# Contribution workflow and the review process

This document outlines the best practice, using git and CI, which
**must** be followed for all contributions to janus-core. Also contained
are instructions and tips for managing your fork of the project which
will help keep merges clean and avoid many headaches.

## Golden rules

In brief the rules for contribution are as follows,

-   Follow the branch, fix, merge model, from your own fork or fork/branch model.
-   An issue must be created for every piece of work (bug, feature,
    *etc.*)
-   Pull/merge requests will not be accepted without a review
-   New features must have a test
-   All tests must pass, no regressions may be merged

## Issues

### Using issues

-   Open an issue for each piece of work done.
-   Open issues for design discussions. The answers may be important for
    newer members.
-   New features, *e.g.* add new MLIP, the comments shall be used to
    provide succint reports on progress.

### Labels

Labels may be assigned to issues to help classify them. Examples
include,

-   Testing: as in testing the water. This label shall be attached to
    things one may think to make one day.
-   Development: this is what I am working now on.
-   Bug: anything in this is critical and shall be fixed asap.
-   Design: queries or suggestions about the structure of the program.
-   Enhancement: for issues relating to new features.

## Review

All merge requests will be reviewed to ensure the integrity of the code.

The reviewer/s have the following responsibilities,
- Ensuring all contribution rules have been followed
- Ensuring the [coding style](./coding_style.md) is adhered to
- Only accepting a merge if all tests have passed
- Using the comments system to request changes for the submittor to make

## Using the git for development

The Github instance hosts an *upstream* repository, which we will refer to
as *upstream*. Contributors will work on their own personal copies of the
repository by creating *forks*. This allows us to keep *upstream* clean
(one commit per merge request, if possible, all commits passing tests)
while users may work on their own *fork*, creating commits and changing
the code as they see fit. Only in exceptional circumstances branches are allowed in *upstream*,
chat in zulip channel about them before you push any.

The *upstream* repository may be cloned as follows,

``` sh
git clone git@github.com:stfc/janus-core.git janus-core-stfc
```

A *fork* is created using the web UI. It may then be cloned for a user
called 'username' as follows,

``` sh
git clone git@github.com:username/janus-core.git janus-core-username
```

### Branch, fix, merge model:

All work should follow the workflow of branch, fix, merge. Let us assume
you have an issue with your code which needs to be fixed.

#### Step 1: Branch from your fork

Create a new branch for the issue on the dashboard of your fork, we will
assume the branch is called 'issueXYZ'. Clone the branch,

``` sh
$ git clone -b issueXYZ --single-branch git@github.com:username/janus-core.git janus-issueXYZ
```

Alternatively you can create the branch in the cli using

``` sh
# clone the repository, if you already have a local repository this is not nessecary
$ git clone git@github.com:username/janus-core.git janus-issueXYZ
$ pushd janus-issueXYZ
# create and checkout a new branch (this is equivilent to git branch followed by git checkout)
$ git checkout -b issueXYZ
# create a remote tracking branch for you to push your changes to
$ git push -u origin issueXYZ
```

#### Step 2: Fix the issue and commit your changes

Fix whatever is wrong. Use git status to see which files you have
changed and prepare a commit.

``` sh
# stage changes
$ git add <filename|folder> to add the new things
# commit the changes with a clear and brief message
$ git commit -m "<commit message>"
# push the commit to origin
$ git push
```

#### Step 3a: Merge your branch into upstream

Follow the provided link in after the push and continue the web interface
Add any relevant labels or milestones and
assign a reviewer. Compare the code and if you are happy click to submit your pull request.

After the pull request has been submitted tests will be run and your
reviewer will be notified.

#### Step 3b: Finalising the merge

If all is OK with the commit your reviewer may set the request to be
merged once all tests pass. Otherwise the reviewer may open discussions
using the Github comment system to point out issues that may need to be
addressed before the commit can be merged.

If changes need to be made you may make more commits onto your branch.
When you push your branch to your *fork* the pull request will be
automatically updated to use the latest commit. Reply to the discussions
to indicate when and how they have been addressed.

If your branch has become out of sync with *upstream* then conflicts may
arise. Sometimes these cannot be automatically resolved and you will
need to resolve them by hand. Github provides instructions for this, or
you can follow this routine,

``` sh
# add upstream as a remote if you have not already
$ git remote add upstream git@github.com:stfc/janus-core.git
# get the changes to upstream since you started working on your issue
$ git fetch upstream
# merge these changes into your branch (assuming you want to merge into the master branch on devel)
$ git merge upstream/main
# resolve any conflicts
# push to your fork
$ git push
```

Alternatively you may use rebase which will replay the changes you made
in your branch on top of *upstream/main* however be sure you understand
the differences between merge and rebase

``` sh
# add upstream as a remote if you have not already
$ git remote add upstream git@github.com:stfc/janus-core.git
# get the changes to upstream since you started working on your issue
$ git fetch devel
# merge these changes into your branch (assuming you want to merge into the master branch on devel)
$ git rebase -i upstream/main
# resolve any conflicts
# push to your fork
$ git push
```

### Advanced git

#### Keeping your fork in sync with project

By adding two remotes, one for *devel* and one for your *fork* it is
possible to keep your *fork* in sync with *devel*. This will greatly
simplify merge requests.

``` sh
# clone your fork
$ git clone git@github.com:username/janus-core.git janus-core-username
pushd janus-core-username
# add a remote for upstream
$ git remote add upstream git@github.com:stfc/janus-core.git
```

These commands need to be done only once. `git remote -v` shall show you
the origin and project fetch and push links

``` sh
$ git remote -v
origin  git@github.com:username/janus-core.git (fetch)
origin  git@github.com:username/janus-core.git (push)
devel git@github.com:stfc/janus-core.git (fetch)
devel git@github.com:stfc/janus-core.git (push)
```

When you need to sync your *fork* with *devel*, do the following,

``` sh
# get the latest commits from upstream
$ git fetch upstream
# ensure you are in the master branch of your fork
$ git checkout main
# merge your main branch into the main branch of upstream
$ git merge upstream/main
# push these changes back to the remote of your fork (origin)
$ git push
```

of course one can use a similar process to merge any other branch or
available projects.

#### Rebasing commits

When working on an issue you may use multiple commits. When you are
ready to create a merge request, you should squash your changes into one
commit in order to keep *devel* clean. This is most easily achieved with
an interactive rebase.

Assuming you have made five commits,

``` sh
# rebase your branch five commits before HEAD i.e. where your branch originally diverged
$ git rebase -i HEAD~5
# follow the instructions. 'pick' the first commit then 'sqaush' or 'fixup' the rest.
# You should now be left with a single commit containing all your changes
# Push your commmit to the remote, use --force-with-lease if you have already pushed this branch to
# 'rewrite history'
$ git push origin branchname --force-with-lease
```

using force is a powerful and dangerous option, use it only if you know
150% nobody else touched that branch.

#### Cleaning stale branches

Deleting branches from the web interface will get rid of the remotes and
not of your local copies. The local branches left behind are called
stale branches. To get rid of them

``` sh
$ git remote prune origin
```

To delete a local branch

``` sh
$ git branch -d localBranch
```

if unmerged commits exists but you still want to delete use

``` sh
$ git branch -D localBranch
```

To delete a remote branch on the remote *origin* use

``` sh
$ git push -d origin remoteBranch
```

## Code Coverage

Ensure that any code you add does not reduce the code coverage in a meaningful way. Reviwers may insist on new test to be added,
please cooperate.