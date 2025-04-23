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
# Using the Pipeline
## Fetching Observations From Remote Machine

Two scripts are available for retrieving image observations from Habitat-Sim environments:

- `ssh_explore_hm3d.py`: Simulates a robot exploring the environment one step at a time.
- `ssh_extract_hm3d.py`: Extracts multiple image observations uniformly or randomly across the environment.

You can specify different scenes using the `--env` command-line argument.

```sh
# Example usage for sampling observations:
python ssh_extract_hm3d.py --env <env-name>
```

This command will generate a file named `semantic_map_<env-name>.pkl` in your local directory.
The file contains a pickled SemanticMap instance representing the collected observations.

## Semantic Segmentation

The segmentation script ssh_oneformer.py requires the previously generated semantic_map_<env-name>.pkl file to be present in your local folder.

It processes the image observations using semantic segmentation and updates the .pkl file with the new results.

```sh 
# Example usage for semantic segmentation:
python ssh_oneformer.py` --env <env-name>
```

⚠️ Note: Ensure that <env-name> matches the environment name used when generating the semantic_map_<env-name>.pkl file.

## 
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