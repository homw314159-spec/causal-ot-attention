# Publishing this source snapshot

1. Review the source files, license, citation metadata and reproduction commands.
2. Create a new repository with an empty history. Do not publish the development
   repository by changing its visibility, or push its existing branches and tags.
3. Commit this snapshot and run its CPU CI. Add the paper's arXiv URL to the README
   and `CITATION.cff` when available; do not invent an identifier in advance.
4. Publish the reviewed initial commit and tag a version, for example `v0.1.0`.
5. Create a release containing the source snapshot and validation notes. An archival
   service such as Zenodo can give that exact version a persistent DOI.
6. Update the paper's code-availability statement with the public repository URL and
   the version or archive DOI. Keep future code changes distinguishable from this release.

The paper submission license and software MIT license are separate choices. Training
data and checkpoints are not part of this source release. The archive should contain
only tracked source files, never `.git`, `.venv`, generated runs, credentials or local
editor settings. Report which checks passed and which GPU/large-scale checks were not
rerun for the release.
