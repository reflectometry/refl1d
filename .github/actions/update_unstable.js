module.exports = async ({github, context, glob}) => {
  const { SEARCH_PATTERN } = process.env;
  console.log({ SEARCH_PATTERN });
  const fs = require('fs');
  const path = require('path');
  const { owner, repo } = context.repo;
  let sid_release = await github.rest.repos.getReleaseByTag({
    owner,
    repo,
    tag: "sid"
  });
  await github.rest.repos.updateRelease({
    owner,
    repo,
    release_id: sid_release.data.id,
    body: "A persistent prerelease where build artifacts for the current tip will be deposited\n\n## Last updated: " + (new Date()).toDateString()
  });
  // delete existing release assets (if needed) and upload new ones:
  const globber = await glob.create(SEARCH_PATTERN ?? "*", {followSymbolicLinks: false});
  const full_paths = await globber.glob();
  for (let full_path of full_paths) {
    const fn = path.basename(full_path);
    const stats = fs.statSync(full_path);
    if (!stats.isFile()) {
      continue;
    }
    console.log("updating: ", fn, full_path);
    let asset_id = (sid_release.data.assets.find((a) => (a.name == fn)) ?? {}).id;
    if (asset_id) {
      await github.rest.repos.deleteReleaseAsset({
        owner,
        repo,
        asset_id
      });
    }
    await github.rest.repos.uploadReleaseAsset({
      owner,
      repo,
      release_id: sid_release.data.id,
      name: fn,
      data: await fs.readFileSync(full_path)
    });
  }
}
