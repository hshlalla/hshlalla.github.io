import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const configPath = path.join(rootDir, "_data", "github_sync.json");
const outputPath = path.join(rootDir, "_data", "github_repos.json");
const githubToken = process.env.GITHUB_TOKEN?.trim();

function buildHeaders() {
  return {
    Accept: "application/vnd.github+json",
    "User-Agent": "audio-archive-github-sync",
    "X-GitHub-Api-Version": "2022-11-28",
    ...(githubToken ? { Authorization: `Bearer ${githubToken}` } : {})
  };
}

async function githubRequest(url) {
  const response = await fetch(url, {
    headers: buildHeaders()
  });

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`GitHub API request failed (${response.status}): ${url}\n${details.slice(0, 240)}`);
  }

  return response.json();
}

function stripMarkdown(markdown) {
  return markdown
    .replace(/^---[\s\S]*?---/m, "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^>\s?/gm, "")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\r/g, " ")
    .replace(/\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildExcerpt(markdown, maxLength) {
  const plainText = stripMarkdown(markdown);

  if (!plainText) {
    return "";
  }

  if (plainText.length <= maxLength) {
    return plainText;
  }

  const sliced = plainText.slice(0, maxLength);
  const lastSpace = sliced.lastIndexOf(" ");

  if (lastSpace < Math.floor(maxLength * 0.6)) {
    return `${sliced.trim()}...`;
  }

  return `${sliced.slice(0, lastSpace).trim()}...`;
}

async function fetchReadme(repoFullName, maxLength) {
  const readme = await githubRequest(`https://api.github.com/repos/${repoFullName}/readme`);

  if (!readme?.content) {
    return {
      readme_excerpt: "",
      readme_html_url: ""
    };
  }

  const decoded = Buffer.from(readme.content.replace(/\n/g, ""), readme.encoding || "base64").toString("utf8");

  return {
    readme_excerpt: buildExcerpt(decoded, maxLength),
    readme_html_url: readme.html_url || ""
  };
}

function normalizeRepo(repo, readmeData) {
  return {
    name: repo.name,
    full_name: repo.full_name,
    repo_url: repo.html_url,
    homepage: repo.homepage || "",
    description: repo.description || "",
    language: repo.language || "",
    topics: Array.isArray(repo.topics) ? repo.topics : [],
    stargazers_count: repo.stargazers_count || 0,
    forks_count: repo.forks_count || 0,
    updated_at: repo.updated_at,
    pushed_at: repo.pushed_at,
    default_branch: repo.default_branch || "main",
    visibility: repo.visibility || "public",
    readme_excerpt: readmeData.readme_excerpt,
    readme_html_url: readmeData.readme_html_url
  };
}

async function main() {
  const config = JSON.parse(await readFile(configPath, "utf8"));
  const username = config.username;
  const maxRepos = Number(config.max_repos || 12);
  const includeForks = Boolean(config.include_forks);
  const excludedRepos = new Set(config.exclude_repos || []);
  const readmeExcerptLength = Number(config.readme_excerpt_length || 260);

  if (!username) {
    throw new Error("`username` is required in _data/github_sync.json");
  }

  const repositories = await githubRequest(
    `https://api.github.com/users/${encodeURIComponent(username)}/repos?sort=updated&direction=desc&per_page=100&type=owner`
  );

  const filteredRepos = repositories
    .filter((repo) => repo.visibility === "public")
    .filter((repo) => !excludedRepos.has(repo.name))
    .filter((repo) => includeForks || !repo.fork)
    .filter((repo) => !repo.archived)
    .sort((left, right) => new Date(right.pushed_at).getTime() - new Date(left.pushed_at).getTime())
    .slice(0, maxRepos);

  const items = [];

  for (const repo of filteredRepos) {
    let readmeData = {
      readme_excerpt: "",
      readme_html_url: ""
    };

    try {
      readmeData = await fetchReadme(repo.full_name, readmeExcerptLength);
    } catch (error) {
      console.warn(`[WARN] README sync skipped for ${repo.full_name}: ${error.message}`);
    }

    items.push(normalizeRepo(repo, readmeData));
  }

  const output = {
    generated_at: new Date().toISOString(),
    source: `https://github.com/${username}`,
    username,
    repo_count: items.length,
    items
  };

  await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");

  console.log(`[SUCCESS] Synced ${items.length} repositories into ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
