# Starting the Repo Locally
## 1. Clone the repository
```sh
git clone https://github.com/usermech/mechademy.git
``` 
## 2. Change directory to the cloned repository
```sh
cd mechademy
```
## 3. Verify you are on the correct branch
```sh
git branch
```
## 4. If you are not on the correct branch, switch to it
```sh
git checkout <branch-name>
```
# 🌿 What is the Feature Branch Workflow?

It’s a Git workflow where **each new feature, bug fix, or task is done in its own separate branch**, instead of working directly on `main`. When the work is done, the branch is merged into `main` via a **pull request (PR)**.

---

## 🔁 Typical Flow (Step-by-Step)

Let's say you have a GitHub repo with a `main` branch. Here's how your students or team members should work:

---

### ✅ 1. **Pull the Latest Code**

Always start by updating local `main`:

```
git checkout main git pull origin main
```




---

### 🌱 2. **Create a Feature Branch**

Create a branch for the specific task or feature:

```bash
git checkout -b feature/add-login-form
```


**Naming tip:**  
Use prefixes like `feature/`, `bugfix/`, or `hotfix/` + a short description:

- `feature/user-profile`
    
- `bugfix/login-error`
    
- `hotfix/crash-on-start`
    

---

### 🛠 3. **Work on the Feature**

Make changes, then stage and commit:

```bash
git add my_script.py
git commit -m "Add login form with basic validation"
```

Commit often and write meaningful messages—it helps with reviewing later.

---

### 🚀 4. **Push the Feature Branch**

Push your branch to GitHub:

```bash
git push origin feature/add-login-form
```




---

### 🔀 5. **Create a Pull Request (PR)**

- Go to the GitHub repo
    
- You’ll see a prompt to "Compare & Pull Request"
    
- Open the PR **into `main`**
    
- Add a description of what was done
    
- Request review (probably from you)
    

---

### 👀 6. **Review & Merge**

You (or a reviewer) checks the code, leaves comments, and either:

- **Approves and merges**
    
- Or **asks for changes**
    

**Merging options:**

- **Squash and merge** (cleans up commit history—great for student projects)
    
- **Rebase and merge** (optional for a clean linear history)
    

---

### 🧹 7. **Delete the Feature Branch**

After merging, delete the branch on GitHub (there’s a button). Optionally delete it locally:

```bash
git branch -d feature/add-login-form 