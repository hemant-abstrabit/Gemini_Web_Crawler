import os
import time
import asyncio
import logging
import random
import platform
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import warnings

# Third-party imports
import trafilatura
import aiohttp
from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright
from google import genai
import backoff
from fake_useragent import UserAgent

# Suppress Windows-specific asyncio warnings
if platform.system() == "Windows":
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*ssl.SSLSocket")
    # Set Windows-specific event loop policy
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# =======================
# CONFIGURATION & MODELS
# =======================

@dataclass
class ScrapingConfig:
    """Configuration for the web scraper"""
    gemini_api_key: str = None
    max_concurrent_requests: int = 5
    request_delay: float = 1.0  # seconds between requests
    max_retries: int = 3
    timeout: int = 30
    output_dir: str = "scraped_content"
    enable_js_rendering: bool = True
    respect_robots_txt: bool = True
    user_agents_rotation: bool = True
    max_content_length: int = 10_000_000  # 10MB limit

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    success: bool
    content: Optional[str] = None
    title: Optional[str] = None  # Page title for filename
    error: Optional[str] = None
    status_code: Optional[int] = None
    content_length: Optional[int] = None
    processing_time: Optional[float] = None

class ProductionWebScraper:
    """Production-ready universal web scraper"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_gemini()
        self.setup_user_agents()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright: Optional[Playwright] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._cleanup_done = False
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('web_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config.output_dir).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def setup_gemini(self):
        """Setup Gemini API client with validation"""
        api_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        try:
            self.gemini_client = genai.Client(api_key=api_key)
            self.logger.info("Gemini client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
            
    def setup_user_agents(self):
        """Setup user agent rotation"""
        if self.config.user_agents_rotation:
            try:
                self.ua = UserAgent()
                self.logger.info("User agent rotation enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize UserAgent: {e}. Using default user agent.")
                self.ua = None
        else:
            self.ua = None
            
    def get_safe_filename(self, title: str, url: str) -> str:
        """Generate a safe filename from title or URL"""
        if title and title.strip():
            # Use title, clean it up
            filename = title.strip()
        else:
            # Fallback to URL-based name
            parsed = urlparse(url)
            filename = f"{parsed.netloc}_{parsed.path.replace('/', '_')}"
        
        # Clean filename - remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and remove extra spaces/underscores
        filename = '_'.join(filename.split())[:100]
        filename = filename.strip('_')
        
        # Ensure we have a valid filename
        if not filename:
            filename = f"scraped_{int(time.time())}"
            
        return f"{filename}.txt"
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        if self.ua:
            try:
                return self.ua.random
            except:
                pass
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_browser()
        await self.initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
        # Additional Windows-specific cleanup
        if platform.system() == "Windows":
            await asyncio.sleep(0.1)  # Give Windows time to clean up processes
            try:
                # Force garbage collection
                import gc
                gc.collect()
            except:
                pass
    
    async def initialize_browser(self):
        """Initialize Playwright browser"""
        if not self.config.enable_js_rendering:
            return
            
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-dev-shm-usage', 
                    '--no-sandbox',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--no-first-run',
                    '--disable-default-apps'
                ]
            )
            self.context = await self.browser.new_context(
                user_agent=self.get_random_user_agent(),
                viewport={'width': 1920, 'height': 1080}
            )
            self.logger.info("Browser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def initialize_session(self):
        """Initialize aiohttp session"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            force_close=True,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.get_random_user_agent()}
        )
        self.logger.info("HTTP session initialized")
    
    async def cleanup(self):
        """Cleanup resources properly"""
        if self._cleanup_done:
            return
            
        self._cleanup_done = True
        
        try:
            # Close context first
            if self.context:
                await self.context.close()
                self.context = None
                
            # Close browser
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            # Stop playwright
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
                
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
                
            self.logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
            
        # Windows-specific additional cleanup
        if platform.system() == "Windows":
            try:
                await asyncio.sleep(0.05)  # Brief pause for Windows cleanup
            except:
                pass
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except Exception:
            return False
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def fetch_with_requests(self, url: str) -> Optional[str]:
        """Fetch content using aiohttp (faster for static content)"""
        try:
            headers = {'User-Agent': self.get_random_user_agent()}
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    if len(content) > self.config.max_content_length:
                        raise ValueError(f"Content too large: {len(content)} bytes")
                    return content
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to fetch {url} with requests: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    async def fetch_with_browser(self, url: str) -> Optional[str]:
        """Fetch content using Playwright (for JS-heavy sites)"""
        if not self.context:
            raise RuntimeError("Browser not initialized")
            
        page = None
        try:
            page = await self.context.new_page()
            
            # Set random user agent for this page
            await page.set_extra_http_headers({
                'User-Agent': self.get_random_user_agent()
            })
            
            # Navigate with timeout and wait for network idle
            await page.goto(url, wait_until='networkidle', timeout=self.config.timeout * 1000)
            
            # Wait for potential dynamic content
            await asyncio.sleep(2)
            
            content = await page.content()
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {url} with browser: {e}")
            raise
        finally:
            if page:
                await page.close()
    
    def extract_clean_content(self, html: str, url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract clean main content and title using trafilatura"""
        try:
            # Extract with metadata to get title
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                favor_precision=True,
                url=url,
                with_metadata=True
            )
            
            if extracted:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    title_tag = soup.find('title')
                    if title_tag:
                        title = title_tag.get_text().strip()
                except:
                    pass

                # # Save extracted content for debugging
                # debug_filename = f"trafilatura_extracted_debug_{int(time.time())}.txt"
                # with open(debug_filename, "w", encoding="utf-8") as f:
                #     f.write(extracted)

                return extracted, title
            return None, None
        except Exception as e:
            self.logger.error(f"Content extraction failed: {e}")
            return None, None
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    async def format_with_gemini(self, content: str, url: str) -> str:
        """
        Format content using Gemini API with chunking for large content.
        Ensures no data is lost due to token/input size limits.
        """
        if not content:
            return ""

        try:
            # Define prompt header (Gemini instructions)
            prompt_header = f"""
                Format this web article content for optimal readability while preserving all information:

                Source URL: {url}

                Instructions:
                - Maintain ALL original content and information
                - Preserve document structure and hierarchy
                - Keep lists, tables, and code blocks intact
                - Ensure proper paragraph breaks and formatting
                - Remove any irrelevant navigation or footer elements
                - Output clean, well-structured text

                Content to format:
                """

            # Set safe chunk size (in characters)
            chunk_size = 15000  # safe for Gemini models
            overlap = 200       # optional: overlap between chunks to avoid mid-sentence cutoffs

            # Chunk the input content
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                # Avoid splitting words/sentences abruptly
                if end < len(content):
                    end = content.rfind('.', start, end) + 1 or end
                chunk = content[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap  # include slight overlap for continuity

            formatted_chunks = []
            for idx, chunk in enumerate(chunks):
                prompt = prompt_header + chunk
                self.logger.info(f"Sending chunk {idx+1}/{len(chunks)} to Gemini...")
                
                max_chunk_retries = 5
                for attempt in range(max_chunk_retries):
                    try:
                        response = await asyncio.to_thread(
                            self.gemini_client.models.generate_content,
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        formatted_text = response.text.strip()
                        formatted_chunks.append(formatted_text)
                        await asyncio.sleep(random.uniform(1.0, 2.5))  # throttle between calls
                        break  # success, exit retry loop
                    except Exception as e:
                        error_msg = str(e).lower()
                        self.logger.warning(f"Gemini chunk {idx+1}/{len(chunks)} attempt {attempt+1} failed: {error_msg}")

                        if "429" in error_msg or "too many requests" in error_msg:
                            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                            self.logger.warning(f"Rate limit hit. Waiting {wait_time:.1f}s before retrying...")
                            await asyncio.sleep(wait_time)
                        else:
                            # not a rate limit error — retry briefly, but not aggressively
                            await asyncio.sleep(1.0)

                        if attempt == max_chunk_retries - 1:
                            self.logger.error(f"Max retries exceeded for chunk {idx+1}. Including raw chunk as-is.")
                            formatted_chunks.append(chunk)

            full_result = "\n\n".join(formatted_chunks)
            return full_result

        except Exception as e:
            self.logger.error(f"Gemini formatting failed for {url}: {e}")
            return content  # fallback to unformatted content
    
    async def scrape_single_url(self, url: str) -> ScrapingResult:
        """Scrape a single URL with comprehensive error handling"""
        start_time = time.time()
        
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Validate URL
                if not self.validate_url(url):
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error="Invalid URL format"
                    )
                
                # Check cache first
                self.logger.info(f"Starting to scrape: {url}")
                
                # Add delay to respect rate limits
                await asyncio.sleep(self.config.request_delay)
                
                html = None
                
                # Try simple HTTP request first (faster)
                try:
                    html = await self.fetch_with_requests(url)
                    self.logger.info(f"Fetched {url} with HTTP client")
                except Exception as e:
                    self.logger.warning(f"HTTP fetch failed for {url}: {e}")
                
                # Fallback to browser if HTTP failed or JS rendering is required
                if not html and self.config.enable_js_rendering:
                    try:
                        html = await self.fetch_with_browser(url)
                        self.logger.info(f"Fetched {url} with browser")
                    except Exception as e:
                        self.logger.error(f"Browser fetch failed for {url}: {e}")
                
                if not html:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error="Failed to fetch content"
                    )
                
                # Extract clean content
                clean_content, page_title = self.extract_clean_content(html, url)
                if not clean_content:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error="Failed to extract main content"
                    )
                
                # Format with Gemini
                formatted_content = await self.format_with_gemini(clean_content, url)
                
                processing_time = time.time() - start_time
                self.logger.info(f"Successfully scraped {url} in {processing_time:.2f}s")
                
                return ScrapingResult(
                    url=url,
                    success=True,
                    content=formatted_content,
                    title=page_title,
                    processing_time=processing_time,
                    content_length=len(formatted_content)
                )
                
            except Exception as e:
                self.logger.error(f"Unexpected error scraping {url}: {e}")
                return ScrapingResult(
                    url=url,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time
                )
    
    async def scrape_multiple_urls(self, urls: List[str]) -> List[ScrapingResult]:
        """Scrape multiple URLs concurrently"""
        self.logger.info(f"Starting to scrape {len(urls)} URLs")
        
        tasks = [self.scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ScrapingResult(
                    url=urls[i],
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        # Log summary
        successful = sum(1 for r in processed_results if r.success)
        self.logger.info(f"Scraping completed: {successful}/{len(urls)} successful")
        
        return processed_results
    
    def save_results(self, results: List[ScrapingResult]):
        """Save each URL's content to a separate file"""
        saved_count = 0
        
        for result in results:
            if result.success and result.content:
                try:
                    # Generate filename from title or URL
                    filename = self.get_safe_filename(result.title, result.url)
                    
                    # Ensure filename is unique
                    filepath = Path(self.config.output_dir) / filename
                    counter = 1
                    base_name = filepath.stem
                    while filepath.exists():
                        filepath = Path(self.config.output_dir) / f"{base_name}_{counter}.txt"
                        counter += 1
                    
                    # Write content to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"URL: {result.url}\n")
                        f.write(f"Title: {result.title or 'N/A'}\n")
                        f.write(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Processing time: {result.processing_time:.2f}s\n")
                        f.write(f"Content length: {result.content_length:,} characters\n")
                        f.write("-" * 80 + "\n\n")
                        f.write(result.content)
                    
                    self.logger.info(f"Saved content to: {filepath}")
                    saved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to save content for {result.url}: {e}")
        
        self.logger.info(f"Successfully saved {saved_count} files to {self.config.output_dir}")
        return saved_count
    
    def generate_report(self, results: List[ScrapingResult]) -> Dict:
        """Generate scraping statistics report"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        report = {
            "total_urls": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "total_processing_time": sum(r.processing_time or 0 for r in results),
            "average_processing_time": sum(r.processing_time or 0 for r in successful) / len(successful) if successful else 0,
            "total_content_length": sum(r.content_length or 0 for r in successful),
            "errors": {}
        }
        
        # Count error types
        for result in failed:
            error_type = result.error or "Unknown error"
            report["errors"][error_type] = report["errors"].get(error_type, 0) + 1
        
        return report

# =======================
# USAGE FUNCTIONS
# =======================

async def scrape_urls(urls: List[str], config: ScrapingConfig = None) -> List[ScrapingResult]:
    """Convenience function to scrape multiple URLs"""
    if config is None:
        config = ScrapingConfig()
    
    async with ProductionWebScraper(config) as scraper:
        results = await scraper.scrape_multiple_urls(urls)
        scraper.save_results(results)
        
        # Generate and print report
        report = scraper.generate_report(results)
        print(f"\n{'='*50}")
        print("SCRAPING REPORT")
        print(f"{'='*50}")
        print(f"Total URLs: {report['total_urls']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Total Time: {report['total_processing_time']:.2f}s")
        print(f"Average Time: {report['average_processing_time']:.2f}s")
        print(f"Total Content: {report['total_content_length']:,} characters")
        
        if report['errors']:
            print(f"\nError Summary:")
            for error, count in report['errors'].items():
                print(f"  {error}: {count}")
        
        return results

# =======================
# EXAMPLE USAGE
# =======================

# if __name__ == "__main__":
#     async def main():
#         # Configuration
#         config = ScrapingConfig(
#             gemini_api_key=os.getenv("GEMINI_API_KEY"),  # Set in environment
#             max_concurrent_requests=3,
#             request_delay=1.0,
#             enable_js_rendering=True
#         )
        
#         # URLs to scrape
#         test_urls = [
#             "https://www.infonetica.net/articles/aca-code-of-ethics-2024",
#             # "https://en.wikipedia.org/wiki/Counseling_psychology"
#             # Add more URLs as needed
#         ]
        
#         try:
#             # Scrape URLs
#             results = await scrape_urls(test_urls, config)
            
#             # # Individual usage example
#             # async with ProductionWebScraper(config) as scraper:
#             #     result = await scraper.scrape_single_url("https://example.com")
#             #     if result.success:
#             #         print(f"Content length: {len(result.content)} characters")
#             #     else:
#             #         print(f"Failed: {result.error}")
#         except Exception as e:
#             print(f"Error in main: {e}")
#         finally:
#             # Windows-specific cleanup
#             if platform.system() == "Windows":
#                 await asyncio.sleep(0.1)
    
#     # Run the scraper with proper Windows handling
#     if platform.system() == "Windows":
#         # Use ProactorEventLoop for Windows
#         try:
#             asyncio.run(main())
#         except RuntimeError as e:
#             if "Event loop is closed" in str(e):
#                 # Handle Windows event loop closure issue
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
#                 try:
#                     loop.run_until_complete(main())
#                 finally:
#                     loop.close()
#             else:
#                 raise
#     else:
#         asyncio.run(main())