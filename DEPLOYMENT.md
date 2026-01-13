# Deploying to Streamlit Cloud

This guide walks you through deploying the Research Paper to Podcast application to Streamlit Cloud.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Local Testing](#1-local-testing-recommended-first-step)
- [Deploy to Streamlit Cloud](#2-deploy-to-streamlit-cloud)
- [Known Limitations](#known-limitations-on-streamlit-cloud)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, you need:

### Required Accounts

- [x] **GitHub account** - To host your code
- [x] **[Streamlit Cloud account](https://streamlit.io/cloud)** - Free tier available
- [x] **OpenAI API key** - For podcast script generation ([Get one here](https://platform.openai.com/api-keys))

### Optional Services

These services unlock additional features but are not required for basic functionality:

- [ ] **AssemblyAI API key** - For audio transcription ([Sign up](https://www.assemblyai.com/))
- [ ] **Firecrawl API key** - For web scraping ([Sign up](https://firecrawl.dev/))
- [ ] **Zep Cloud API key** - For conversation memory ([Sign up](https://www.getzep.com/))

---

## Quick Start

### 1. Local Testing (Recommended First Step)

Always test locally before deploying to ensure everything works correctly.

#### Option A: Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/notebook-lm-clone.git
cd notebook-lm-clone

# Install dependencies with uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run the app
uv run streamlit run app.py
```

#### Option B: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/notebook-lm-clone.git
cd notebook-lm-clone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run the app
streamlit run app.py
```

#### Configure Your API Keys

Edit the `.env` file and add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini

# Optional
# ASSEMBLYAI_API_KEY=your-key
# FIRECRAWL_API_KEY=your-key
# ZEP_API_KEY=your-key
```

#### Test the Application

1. Open your browser to `http://localhost:8501`
2. Try uploading a PDF document
3. Generate a podcast script
4. Verify everything works as expected

---

### 2. Deploy to Streamlit Cloud

Once local testing is successful, you're ready to deploy!

#### Step 2.1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Ready for Streamlit deployment"

# Push to GitHub
git remote add origin https://github.com/your-username/notebook-lm-clone.git
git branch -M main
git push -u origin main
```

**Important:** The `.gitignore` file ensures that `.env` and other sensitive files are NOT pushed to GitHub.

#### Step 2.2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"** button
4. Configure your app:
   - **Repository:** Select `your-username/notebook-lm-clone`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** 3.11
5. Click **"Deploy!"**

#### Step 2.3: Configure Secrets

Your app will initially fail because it doesn't have API keys. Let's fix that:

1. In your Streamlit Cloud dashboard, click on your app
2. Go to **Settings** (⚙️ icon) → **Secrets**
3. Add your secrets in TOML format:

```toml
# Required - OpenAI Configuration
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
OPENAI_MODEL = "gpt-4o-mini"

# Optional - Uncomment if you have these services
# ASSEMBLYAI_API_KEY = "your-assemblyai-key"
# FIRECRAWL_API_KEY = "your-firecrawl-key"
# ZEP_API_KEY = "your-zep-key"
# ZEP_PROJECT_ID = "your-project-id"
# ZEP_COLLECTION = "research_papers"
```

4. Click **"Save"**
5. Your app will automatically reboot with the new secrets

#### Step 2.4: Verify Deployment

Wait 2-5 minutes for the app to deploy (first deployment takes longer due to model downloads).

Test your deployed app:
- [ ] Upload a PDF document
- [ ] Add text content
- [ ] Generate a podcast script
- [ ] Check that citations are working
- [ ] (Optional) Try audio generation if Kokoro TTS is enabled

---

## Configuration Files

Your deployment uses these configuration files:

### `.streamlit/config.toml`

Controls how Streamlit runs your app:

```toml
[server]
headless = true
maxUploadSize = 200  # MB
```

### `.streamlitignore`

Excludes files from deployment (reduces upload time):
- Development files (notebooks, tests)
- Large data files
- Build artifacts

### `requirements.txt`

Lists all Python dependencies. Streamlit Cloud installs these automatically.

---

## Known Limitations on Streamlit Cloud

Be aware of these constraints when using the free tier:

### Resource Limits

| Resource | Free Tier Limit | Impact |
|----------|-----------------|--------|
| **RAM** | 1 GB | May struggle with large PDFs or long podcasts |
| **Storage** | Ephemeral | Data lost on app reboot |
| **CPU** | Shared | Slower processing |

### Performance Considerations

- **Cold starts:** First load can take 30-60 seconds while downloading models
- **Model downloads:**
  - FastEmbed: ~100 MB (for embeddings)
  - Kokoro TTS: ~500 MB (for audio generation)
  - These download once per deployment
- **Audio generation:** May fail if TTS models can't load due to memory constraints
- **Large documents:** PDFs over 10 MB may cause memory issues

### Storage Behavior

- The filesystem is **ephemeral** - all uploaded files and generated outputs are lost when the app restarts
- Vector database (Milvus) stores embeddings in memory - data is not persisted between sessions
- No persistent storage for conversation history

---

## Troubleshooting

### App Won't Start

**Symptom:** App shows error on initial load

**Solutions:**
1. Check the logs in Streamlit Cloud dashboard
2. Verify all secrets are correctly formatted (TOML syntax is strict!)
3. Ensure Python version is 3.11 or 3.12
4. Check that all required API keys are added to secrets

**Example of correct TOML formatting:**
```toml
# Good
OPENAI_API_KEY = "sk-proj-abc123"

# Bad (will cause errors)
OPENAI_API_KEY = sk-proj-abc123  # Missing quotes
OPENAI_API_KEY: "sk-proj-abc123"  # Wrong separator (: instead of =)
```

### Memory Errors

**Symptom:** App crashes with "Killed" or memory errors

**Solutions:**
1. Reduce `maxUploadSize` in `.streamlit/config.toml`
2. Process smaller documents (under 5 MB)
3. Disable audio generation to reduce memory usage
4. Consider upgrading to Streamlit Community Cloud ($20/month for 4 GB RAM)

**Quick fix - Disable TTS:**
Set this in your secrets:
```toml
DISABLE_TTS = true
```

### Slow Performance

**Symptom:** App takes a long time to respond

**Expected behavior:**
- **First run:** 30-60 seconds (model downloads)
- **Subsequent runs:** 5-10 seconds (models cached)
- **PDF processing:** 10-30 seconds depending on size

**Solutions:**
1. Wait for models to download on first run
2. Use smaller documents for faster processing
3. Consider pre-building a Docker image with models included (advanced)

### API Key Errors

**Symptom:** "Invalid API key" or "Authentication failed"

**Solutions:**
1. Verify your OpenAI API key is valid and has credits
2. Check that the key is correctly copied (no extra spaces)
3. Ensure the key hasn't been revoked
4. Regenerate a new API key if needed

### Import Errors

**Symptom:** "ModuleNotFoundError" or similar

**Solutions:**
1. Ensure `requirements.txt` is in the root directory
2. Check that all dependencies are listed
3. Verify Python version compatibility (3.11 or 3.12)
4. Try manually rebooting the app

---

## Advanced: Docker Deployment

For self-hosting, you can use Docker. Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t notebook-lm .
docker run -p 8501:8501 --env-file .env notebook-lm
```

---

## Updating Your Deployment

When you make changes to your code:

```bash
# Make your changes
git add .
git commit -m "Your changes"
git push origin main
```

Streamlit Cloud will automatically detect the changes and redeploy your app within 1-2 minutes.

---

## Security Best Practices

1. **Never commit `.env` files** - Always use Streamlit Secrets in production
2. **Rotate API keys regularly** - Especially if you suspect they've been exposed
3. **Use least-privilege API keys** - Create API keys with only the permissions you need
4. **Monitor API usage** - Set up billing alerts in OpenAI dashboard
5. **Review logs regularly** - Check for unusual activity or errors

---

## Getting Help

- **Streamlit Documentation:** https://docs.streamlit.io/
- **Streamlit Community Forum:** https://discuss.streamlit.io/
- **Project Issues:** [GitHub Issues](https://github.com/your-username/notebook-lm-clone/issues)

---

## Next Steps

Once deployed successfully:

1. **Share your app** - Get the public URL from Streamlit Cloud dashboard
2. **Add custom domain** (optional) - Available on paid plans
3. **Monitor usage** - Check API usage in OpenAI dashboard
4. **Gather feedback** - Share with users and iterate
5. **Consider upgrades** - If hitting resource limits, upgrade to paid tier

Happy deploying! 🚀
