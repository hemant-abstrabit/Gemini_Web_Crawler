import streamlit as st
import asyncio
import os
import time
import io
import zipfile
from typing import List
import platform
import warnings
import subprocess
import sys


# Suppress warnings for cleaner UI
if platform.system() == "Windows":
    warnings.filterwarnings("ignore", category=ResourceWarning)

# Page configuration
st.set_page_config(
    page_title="Universal Web Scraper",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Playwright installation for Streamlit Cloud
@st.cache_resource
def install_playwright():
    """Install playwright browsers on first run"""
    try:
        # Check if running on Streamlit Cloud
        if os.getenv("STREAMLIT_SHARING_MODE") or "streamlit" in os.getcwd().lower():
            st.info("🔄 Installing browser dependencies... This may take a moment on first run.")
            
            # Install playwright browsers
            result = subprocess.run([
                sys.executable, "-m", "playwright", "install", "chromium"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                st.success("✅ Browser dependencies installed successfully!")
                return True
            else:
                st.error(f"❌ Failed to install browsers: {result.stderr}")
                return False
        return True
    except Exception as e:
        st.error(f"❌ Browser installation error: {str(e)}")
        return False

# Try to install playwright on startup
playwright_ready = install_playwright()

# Import your scraping classes (make sure this is after playwright installation)
try:
    from final3 import ProductionWebScraper, ScrapingConfig, ScrapingResult
except ImportError as e:
    st.error(f"❌ Failed to import scraper: {str(e)}")
    st.stop()



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .url-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .content-box {
        background-color: #ffffff;
        color: #222222;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .content-box {
            background-color: #2c2c2c;
            color: #f0f0f0;
            border: 1px solid #555555;
        }
        .url-header {
            background-color: #333333;
            border-left-color: #1f77b4;
            color: #e0e0e0;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'scraping_results' not in st.session_state:
        st.session_state.scraping_results = []
    if 'last_scrape_time' not in st.session_state:
        st.session_state.last_scrape_time = None

def validate_gemini_api_key(api_key: str) -> bool:
    """Validate Gemini API key format"""
    return api_key and len(api_key.strip()) > 10

def create_download_content(results: List[ScrapingResult]) -> str:
    """Create combined content for download"""
    download_content = []
    
    for i, result in enumerate(results, 1):
        if result.success and result.content:
            download_content.append(f"{'='*80}")
            download_content.append(f"SCRAPED CONTENT #{i}")
            download_content.append(f"{'='*80}")
            download_content.append(f"URL: {result.url}")
            download_content.append(f"Title: {result.title or 'N/A'}")
            download_content.append(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            download_content.append(f"Processing time: {result.processing_time:.2f}s")
            download_content.append(f"Content length: {result.content_length:,} characters")
            download_content.append(f"{'-'*80}")
            download_content.append("")
            download_content.append(result.content)
            download_content.append("\n\n")
    
    return "\n".join(download_content)

def create_individual_download(result: ScrapingResult) -> str:
    """Create content for individual URL download"""
    if not result.success or not result.content:
        return ""
    
    download_content = []
    download_content.append(f"URL: {result.url}")
    download_content.append(f"Title: {result.title or 'N/A'}")
    download_content.append(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    download_content.append(f"Processing time: {result.processing_time:.2f}s")
    download_content.append(f"Content length: {result.content_length:,} characters")
    download_content.append(f"{'-'*80}")
    download_content.append("")
    download_content.append(result.content)
    
    return "\n".join(download_content)

async def scrape_urls_async(urls: List[str], config: ScrapingConfig) -> List[ScrapingResult]:
    """Async wrapper for scraping URLs with Streamlit Cloud compatibility"""
    try:
        # For Streamlit Cloud, reduce resource usage
        if os.getenv("STREAMLIT_SHARING_MODE") or "streamlit" in os.getcwd().lower():
            config.max_concurrent_requests = min(config.max_concurrent_requests, 2)
            config.request_delay = max(config.request_delay, 2.0)
            config.timeout = min(config.timeout, 30)
        
        async with ProductionWebScraper(config) as scraper:
            results = await scraper.scrape_multiple_urls(urls)
            return results
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []

def run_scraping(urls: List[str], config: ScrapingConfig) -> List[ScrapingResult]:
    """Run scraping with proper asyncio handling"""
    try:
        # Handle Windows event loop issues
        if platform.system() == "Windows":
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        
        return asyncio.run(scrape_urls_async(urls, config))
    except Exception as e:
        st.error(f"Error running scraper: {str(e)}")
        return []

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🕷️ Universal Web Scraper</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key",
            placeholder="AIza..."
        )
        
        # Scraping settings
        st.subheader("Scraping Settings")
        
        max_concurrent = st.slider(
            "Max Concurrent Requests",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of URLs to scrape simultaneously"
        )
        
        request_delay = st.slider(
            "Request Delay (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Delay between requests to be respectful to servers"
        )
        
        enable_js = st.checkbox(
            "Enable JavaScript Rendering",
            value=True,
            help="Use browser to render JavaScript-heavy pages (slower but more comprehensive)"
        )
        
        timeout = st.slider(
            "Timeout (seconds)",
            min_value=10,
            max_value=120,
            value=30,
            help="Maximum time to wait for page load"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter URLs to Scrape")
        
        # URL input options
        input_method = st.radio(
            "Input Method:",
            ["Single URL", "Multiple URLs (one per line)"],
            horizontal=True
        )
        
        if input_method == "Single URL":
            url_input = st.text_input(
                "Enter URL:",
                placeholder="https://example.com",
                help="Enter a single URL to scrape"
            )
            urls = [url_input.strip()] if url_input.strip() else []
        else:
            url_input = st.text_area(
                "Enter URLs (one per line):",
                height=150,
                placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
                help="Enter multiple URLs, one per line"
            )
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        # Validation and scraping button
        valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        
        if urls and len(valid_urls) != len(urls):
            st.warning(f"⚠️ {len(urls) - len(valid_urls)} invalid URL(s) detected. Only valid HTTP/HTTPS URLs will be scraped.")
        
        # Scrape button
        if st.button("🚀 Start Scraping", type="primary", disabled=not valid_urls or not playwright_ready):
            if not playwright_ready:
                st.error("❌ Browser dependencies not ready. Please refresh the page and try again.")
            elif not validate_gemini_api_key(gemini_api_key):
                st.error("❌ Please enter a valid Gemini API key in the sidebar.")
            else:
                # Create configuration
                config = ScrapingConfig(
                    gemini_api_key=gemini_api_key,
                    max_concurrent_requests=max_concurrent,
                    request_delay=request_delay,
                    enable_js_rendering=enable_js,
                    timeout=timeout
                )
                
                # Show progress
                with st.spinner(f"Scraping {len(valid_urls)} URL(s)... This may take a while."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress (simulated for now)
                    for i in range(len(valid_urls)):
                        status_text.text(f"Processing URL {i+1}/{len(valid_urls)}")
                        progress_bar.progress((i+1)/len(valid_urls))
                    
                    # Run scraping
                    results = run_scraping(valid_urls, config)
                    
                    # Store results in session state
                    st.session_state.scraping_results = results
                    st.session_state.last_scrape_time = time.strftime('%Y-%m-%d %H:%M:%S')
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Scraping completed! Check results below.")
    
    with col2:
        st.header("📊 Statistics")
        
        if st.session_state.scraping_results:
            results = st.session_state.scraping_results
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            # Stats cards
            st.metric("Total URLs", len(results))
            st.metric("Successful", successful)
            st.metric("Failed", failed)
            
            if successful > 0:
                success_rate = (successful / len(results)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
                
                avg_time = sum(r.processing_time or 0 for r in results if r.success) / successful
                st.metric("Avg Time", f"{avg_time:.1f}s")
            
            st.info(f"Last scraped: {st.session_state.last_scrape_time}")
        else:
            st.info("No scraping results yet. Enter URLs and click 'Start Scraping' to begin.")
    
    # Results section
    if st.session_state.scraping_results:
        st.markdown("---")
        st.header("📄 Scraping Results")
        
        results = st.session_state.scraping_results
        successful_results = [r for r in results if r.success]
        
        # Download all button
        if successful_results:
            all_content = create_download_content(successful_results)
            st.download_button(
                label="📥 Download All Content",
                data=all_content,
                file_name=f"scraped_content_{int(time.time())}.txt",
                mime="text/plain",
                type="secondary"
            )
        
        # Display results for each URL
        for i, result in enumerate(results, 1):
            with st.container():
                # URL header with status
                if result.success:
                    st.markdown(f'<div class="url-header"><strong>#{i} - ✅ SUCCESS</strong><br><a href="{result.url}" target="_blank">{result.url}</a></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="url-header"><strong>#{i} - ❌ FAILED</strong><br><a href="{result.url}" target="_blank">{result.url}</a></div>', unsafe_allow_html=True)
                
                if result.success and result.content:
                    # Success case
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if result.title:
                            st.write(f"**Title:** {result.title}")
                    
                    with col2:
                        st.write(f"**Length:** {result.content_length:,} chars")
                    
                    with col3:
                        st.write(f"**Time:** {result.processing_time:.1f}s")
                    
                    # Content display
                    with st.expander(f"📖 View Content (Click to expand)", expanded=False):
                        st.markdown(f'<div class="content-box">{result.content[:2000]}{"..." if len(result.content) > 2000 else ""}</div>', unsafe_allow_html=True)
                    
                    # Individual download button
                    individual_content = create_individual_download(result)
                    filename = f"scraped_{i}_{int(time.time())}.txt"
                    
                    st.download_button(
                        label=f"📥 Download Content #{i}",
                        data=individual_content,
                        file_name=filename,
                        mime="text/plain",
                        key=f"download_{i}"
                    )
                
                else:
                    # Error case
                    st.markdown(f'<div class="error-message"><strong>Error:</strong> {result.error}</div>', unsafe_allow_html=True)
                
                st.markdown("---")

if __name__ == "__main__":
    main()