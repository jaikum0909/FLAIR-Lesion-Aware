# How to Upload This Folder to GitHub

Follow these steps on your **personal laptop** after downloading this folder from Google Drive.

---

## Step 1 — Create a new GitHub repository

1. Go to https://github.com/new
2. Repository name: `FLAIR-Lesion-Aware`
3. Set to **Public**
4. **Do NOT** tick "Add a README file" (we already have one)
5. Click **Create repository**
6. Copy the repo URL shown — it will look like:
   `https://github.com/YOUR-USERNAME/FLAIR-Lesion-Aware.git`

---

## Step 2 — Open Terminal and run these commands

```bash
# 1. Go into the folder (adjust path to where you downloaded it)
cd ~/Downloads/FLAIR-Lesion-Aware

# 2. Initialise git
git init

# 3. Stage everything
git add .

# 4. First commit
git commit -m "Lesion-Aware Vision-Language Learning for Retinal Disease Diagnosis

M.Tech Thesis — IIIT Allahabad (MHC2024011)
- LoRA adaptation for FLAIR vision encoder (rank-4, layer3+layer4)
- Lesion-Aware Cross-Attention with 16 clinical lesion queries
- Training scripts, Grad-CAM visualisation, full ablation results"

# 5. Connect to your GitHub repo (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/FLAIR-Lesion-Aware.git

# 6. Push
git branch -M main
git push -u origin main
```

---

## Step 3 — Share the link

Once pushed, your repo will be live at:
`https://github.com/YOUR-USERNAME/FLAIR-Lesion-Aware`

Share this link with your guide.

---

## Note on datasets

The `local_data/data/` folder (raw images) is **not included** in this upload — it's too large (~several GB).
The CSV split files (`local_data/dataframes/`) **are included** and are needed to reproduce training.
To re-run experiments, place the downloaded datasets under `local_data/data/` following the structure in `local_data/constants.py`.
