#!/usr/bin/env python3
"""
Deployment helper script for Streamlit Community Cloud
Run this before deploying to ensure all dependencies are properly configured
"""

import subprocess
import sys
import os
from pathlib import Path

def create_requirements_txt():
    """Create requirements.txt with all necessary packages"""
    requirements = [
        "streamlit>=1.28.0",
        "aiohttp>=3.8.0", 
        "playwright>=1.40.0",
        "trafilatura>=1.6.0",
        "google-genai>=1.19.0",
        "google-generativeai>=0.3.0",
        "backoff>=2.2.0",
        "fake-useragent>=1.4.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "requests>=2.28.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("✅ Created requirements.txt")

def create_packages_txt():
    """Create packages.txt for system dependencies"""
    packages = [
        "fonts-liberation",
        "libasound2",
        "libatk-bridge2.0-0",
        "libdrm2",
        "libgtk-3-0",
        "libxkbcommon0",
        "libxrandr2",
        "libxss1",
        "libnss3"
    ]
    
    with open("packages.txt", "w") as f:
        f.write("\n".join(packages))
    
    print("✅ Created packages.txt")

def create_streamlit_config():
    """Create Streamlit configuration"""
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    config_content = """[global]
developmentMode = false

[server]
headless = true
enableCORS = false
port = 8501
maxUploadSize = 50

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    with open(config_dir / "config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Created .streamlit/config.toml")

def create_secrets_template():
    """Create secrets template for Streamlit Cloud"""
    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    
    secrets_content = """# Add your Gemini API key here (for local development)
# For Streamlit Cloud, add this in the app dashboard secrets section
GEMINI_API_KEY = "your-gemini-api-key-here"
"""
    
    with open(secrets_dir / "secrets.toml", "w") as f:
        f.write(secrets_content)
    
    print("✅ Created .streamlit/secrets.toml template")

def create_readme():
    """Create deployment README"""
    readme_content = """# Web Scraper Streamlit App

## Deployment Instructions for Streamlit Community Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy web scraper"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add Secrets**:
   - In Streamlit Cloud dashboard, go to your app
   - Click "Settings" → "Secrets"
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your-actual-api-key-here"
     ```

4. **Wait for Build**:
   - First deployment takes 5-10 minutes
   - Browser dependencies will install automatically
   - App will be available at: `https://your-app-name.streamlit.app`

## Files Required for Deployment:
- ✅ app.py (main application)
- ✅ final3.py (scraper module)
- ✅ requirements.txt (Python dependencies)
- ✅ packages.txt (system dependencies)
- ✅ .streamlit/config.toml (Streamlit configuration)

## Testing Locally:
```bash
streamlit run app.py
```

## Troubleshooting:
- If deployment fails, check the logs in Streamlit Cloud dashboard
- Ensure all files are pushed to GitHub
- Verify API key is correctly set in secrets
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def main():
    """Run all deployment preparation steps"""
    print("🚀 Preparing for Streamlit Community Cloud deployment...\n")
    
    # Create all necessary files
    create_requirements_txt()
    create_packages_txt()
    create_streamlit_config()
    create_secrets_template()
    create_readme()
    
    print("\n🎉 Deployment preparation complete!")
    print("\nNext steps:")
    print("1. Add your Gemini API key to .streamlit/secrets.toml (for local testing)")
    print("2. Test locally: streamlit run app.py")
    print("3. Push to GitHub: git add . && git commit -m 'Deploy' && git push")
    print("4. Deploy on Streamlit Cloud and add API key to secrets")
    print("\n📖 See README.md for detailed instructions")

if __name__ == "__main__":
    main()