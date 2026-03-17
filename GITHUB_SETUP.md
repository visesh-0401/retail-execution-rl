# GitHub Setup: Step-by-Step (10 minutes)
**Goal**: Create repo and push your Month 1 code  
**Time**: ~10 min  
**Deadline**: Do THIS TODAY before anything else

---

## Step 1: Create GitHub Repo (2 min)

1. Go to **https://github.com/new**
2. Fill in:
   - **Repository name**: `retail-execution-rl`
   - **Description**: "Reinforcement Learning for Retail Order Execution Under API Rate Limits"
   - **Visibility**: Public (so you can cite it)
   - **Add .gitignore**: Python
   - **Add README**: (optional, we have one)
   - **License**: MIT

3. Click **"Create repository"**

4. **You'll see**: "Quick setup" page with commands. Keep this open! 👇

---

## Step 2: Push Your Code (8 min)

In PowerShell on your local machine:

```powershell
cd d:\Project

# Step 1: Check git is set up
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"

# Step 2: Initialize if not already done (skip if already initialized)
# git init

# Step 3: Add all files
git add -A

# Step 4: Initial commit
git commit -m "Month 1 Complete: Simulator + Baselines + Data ready"

# Step 5: Connect to GitHub (COPY THIS FROM YOUR GitHub REPO PAGE)
# Replace YOUR_USERNAME with actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/retail-execution-rl.git

# Step 6: Name branch main and push
git branch -M main
git push -u origin main

# Expected output: 
# Enumerating objects: XX done.
# Compressing objects: 100% (XX/XX) done.
# Writing objects: 100% (XX/XX) done.
# ...
# * [new branch]      main -> main
# Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Step 3: Verify (1 min)

Go to **https://github.com/YOUR_USERNAME/retail-execution-rl**

You should see:
- ✅ README.md
- ✅ requirements.txt
- ✅ src/ folder
- ✅ scripts/ folder
- ✅ data/ folder
- ✅ configs/ folder
- ✅ kaggle/ folder

If all there: **SUCCESS!** 🎉

---

## Section 2: Create GitHub Access Token (for if you get auth errors)

If `git push` fails with authentication error:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" > "Generate new token (classic)"
3. Name: `retail-execution-rl-pushes`
4. Select: `repo` (full control of repos)
5. Expiration: 90 days (fine for this project)
6. Click "Generate token"
7. **COPY the token** (long string of characters)

8. In PowerShell, when it asks for password:
   - Paste the token instead of your password
   - Done!

---

## Verify Your Setup

After pushing, run this to confirm:

```powershell
cd d:\Project

# Should show: https://github.com/YOUR_USERNAME/retail-execution-rl.git
git remote -v

# Should show: 10+ commits
git log --oneline | head -10
```

---

## Total Checklist

- [ ] Created GitHub repo (retail-execution-rl)
- [ ] Set user.name and user.email
- [ ] Ran `git add -A`
- [ ] Ran `git commit`
- [ ] Ran `git remote add origin`
- [ ] Ran `git push -u origin main`
- [ ] Verified repo on GitHub (can see folders)

**If all checked**: You're ready for Week 6! ✅

---

## Next: Week 5 Tasks (After GitHub is done)

Once GitHub is pushed:

1. ✅ Verify config loads (2 min)
   ```bash
   python -c "import yaml; c = yaml.safe_load(open('configs/training_config.yaml')); print(c['training']['timesteps'])"
   ```

2. ✅ Print WEEK_6_EXECUTION_CHECKLIST.md (1 min)

3. ✅ Test Kaggle GPU (5 min)
   - Log into Kaggle
   - Create new notebook
   - Enable GPU
   - Run: `import torch; print(torch.cuda.get_device_name(0))`

---

## Questions?

If GitHub push fails:
- **Auth error**: Use personal access token (see Section 2)
- **Remote rejected**: Check you typed username correctly
- **Can't find git**: Install Git for Windows (git-scm.com)

---

**Priority: Finish GitHub TODAY. Then everything else flows smoothly.**

Go! 🚀

