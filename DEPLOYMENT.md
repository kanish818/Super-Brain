# Deployment Guide - Brain-Controlled Wheelchair Predictor

This guide covers deploying the Brain-Controlled Wheelchair web application to cloud platforms.

## Prerequisites
- GitHub repository: https://github.com/kanish818/Super-Brain
- Model file: `Project/predict/dqn_wheelchair_model.keras` (must be in repo)
- Data file: `Final_clean.csv` (must be in repo)

## Option 1: Deploy to Render (Recommended - Free Tier Available)

### Steps:
1. **Push your code to GitHub** (make sure all files are committed)
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push origin main
   ```

2. **Sign up/Login to Render**
   - Go to https://render.com
   - Sign up with your GitHub account

3. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `kanish818/Super-Brain`
   - Configure:
     - **Name**: `brain-wheelchair-predictor`
     - **Region**: Choose closest to you
     - **Branch**: `main`
     - **Root Directory**: Leave empty (or `.`)
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn webapp:app --bind 0.0.0.0:$PORT --timeout 120`
     - **Instance Type**: Free (or paid for better performance)

4. **Environment Variables** (Optional)
   - Add these if needed:
     - `PYTHON_VERSION` = `3.13.5`
     - `TF_ENABLE_ONEDNN_OPTS` = `0` (reduces TensorFlow warnings)

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build and deployment
   - Your app will be live at: `https://brain-wheelchair-predictor.onrender.com`

### Notes:
- Free tier may sleep after inactivity (first request takes ~30s to wake)
- For production, upgrade to paid tier for always-on service

---

## Option 2: Deploy to Railway

### Steps:
1. **Push code to GitHub** (if not done already)

2. **Sign up/Login to Railway**
   - Go to https://railway.app
   - Sign up with GitHub

3. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `kanish818/Super-Brain`

4. **Configure**
   - Railway auto-detects Python
   - It will use `Procfile` automatically
   - Set environment variables if needed:
     - `PORT` is auto-set by Railway
     - `PYTHON_VERSION` = `3.13.5`

5. **Deploy**
   - Railway builds and deploys automatically
   - Get your URL from the deployment dashboard

---

## Option 3: Deploy to Heroku

### Steps:
1. **Install Heroku CLI**
   ```bash
   # Windows (PowerShell as Admin)
   winget install Heroku.HerokuCLI
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   cd "C:\Resumes\NeenOpal\Super Brain"
   heroku create brain-wheelchair-app
   ```

4. **Set Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

6. **Open App**
   ```bash
   heroku open
   ```

### Notes:
- Heroku free tier was discontinued; requires paid plan
- Good for production with add-ons (monitoring, logging, etc.)

---

## Option 4: Deploy with Docker (Any Cloud Platform)

### Create Dockerfile:
See `Dockerfile` in the repository.

### Build and Run Locally:
```bash
docker build -t brain-wheelchair .
docker run -p 5000:5000 brain-wheelchair
```

### Deploy to:
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **Azure Container Instances**
- **DigitalOcean App Platform**

---

## Post-Deployment Checklist

### 1. Test the Deployment
- Visit your deployed URL
- Upload `sample_input.csv` or enter manual values: `0.5, -0.2, 0.8`
- Verify predictions work

### 2. Monitor Logs
- **Render**: Dashboard → Logs tab
- **Railway**: Project → Deployments → Logs
- **Heroku**: `heroku logs --tail`

### 3. Set Up Custom Domain (Optional)
- Most platforms allow free custom domain linking
- Example: `wheelchair.yourdomain.com`

### 4. Enable HTTPS
- All platforms provide free SSL/TLS certificates automatically

---

## Troubleshooting

### Issue: "Application Error" or 500 Error
**Solution**: Check logs for missing files or import errors
```bash
# Heroku
heroku logs --tail

# Render/Railway
Check dashboard logs
```

### Issue: TensorFlow Import Fails
**Solution**: Ensure `tensorflow>=2.13.0` is in `requirements.txt`

### Issue: Model File Not Found
**Solution**: 
- Verify `Project/predict/dqn_wheelchair_model.keras` is committed to Git
- Check `.gitignore` doesn't exclude `.keras` files

### Issue: Slow First Request
**Solution**: 
- Free tiers sleep after inactivity
- Upgrade to paid tier for always-on
- Or use a ping service to keep app awake

### Issue: Out of Memory
**Solution**:
- TensorFlow + model can be large
- Upgrade to instance with more RAM (512MB minimum, 1GB recommended)

---

## Performance Optimization

### 1. Add Model Caching
Already implemented - model loads once at startup

### 2. Use Production WSGI Server
Already configured - using `gunicorn`

### 3. Enable Gzip Compression
Add to `webapp.py`:
```python
from flask_compress import Compress
Compress(app)
```

### 4. Add Health Check Endpoint
Already available at: `/` (root)

---

## Cost Estimates

| Platform | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Render** | 750 hrs/month (sleeps) | $7/month (always-on) |
| **Railway** | $5 credit/month | Pay-as-you-go (~$5-10/mo) |
| **Heroku** | None | $7/month (Eco dyno) |
| **Fly.io** | 3 VMs free | ~$5/month |

---

## Recommended Setup for Production

1. **Platform**: Render or Railway (easiest)
2. **Tier**: Paid ($7-10/month for reliability)
3. **Monitoring**: Enable platform monitoring + add Sentry for error tracking
4. **Backups**: Keep model files in Git LFS or cloud storage
5. **CI/CD**: Platforms auto-deploy on git push

---

## Quick Start (Render - Fastest)

1. Commit and push:
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push origin main
   ```

2. Go to https://render.com → New Web Service

3. Connect repo → Use these settings:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn webapp:app --bind 0.0.0.0:$PORT`

4. Deploy → Wait 5 min → Done! 🚀

Your app will be live at: `https://[your-app-name].onrender.com`
