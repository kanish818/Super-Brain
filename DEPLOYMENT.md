# Deployment Guide

## Quick Deploy Options

### Option 1: Render (Recommended - Free tier available)
1. Go to https://render.com/
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repo: `kanish818/Super-Brain`
5. Configure:
   - **Name**: super-brain-wheelchair
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn webapp:app`
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Wait 5-10 minutes for deployment

### Option 2: Railway
1. Go to https://railway.app/
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `kanish818/Super-Brain`
5. Railway auto-detects Python and deploys automatically
6. Your app will be live at the provided URL

### Option 3: Heroku
```bash
# Install Heroku CLI first
heroku login
heroku create super-brain-wheelchair
git push heroku main
heroku open
```

## Environment Variables (if needed)
- `PORT` - Auto-set by most platforms
- `PYTHON_VERSION` - Set to 3.11.9

## Post-Deployment
After deployment, visit `https://your-app-url.com` and test with:
- Manual input: `0.5, -0.2, 0.8`
- Or upload `sample_input.csv`

## Troubleshooting
- **Build fails**: Check build logs for missing dependencies
- **App crashes**: Ensure all CSV files (`Final_clean.csv`, model file) are in repo
- **Slow startup**: TensorFlow takes 30-60s to load on first request
