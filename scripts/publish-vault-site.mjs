#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_VAULT_PATH = 'G:\\My Drive\\Bobsidian\\github.vault';
const DEFAULT_EXPORT_PATH = 'G:\\My Drive\\Bobsidian\\seihwanMoon.github.io';
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

const ROOT_PAGE_ORDER = [
  'Home.md',
  'Topics.md',
  'Updates.md',
  'Now.md',
  'About.md',
  'Garden-Structure.md',
  'index.md',
  'log.md',
  'AGENTS.md'
];

const ROOT_FOLDER_ORDER = ['topics', 'updates'];
const TOPICS_FOLDER_ORDER = [
  'coding-systems',
  'ai-agents',
  'llm-platforms',
  'automation',
  'ml-dl',
  'industrial-automation',
  'notes-and-methods'
];
const LLM_SUBFOLDER_ORDER = ['codex', 'claude', 'openclaw', 'comparisons'];

const FIELD_NAMES = ['title', 'aliases', 'headers', 'tags', 'path', 'content'];
const FIELD_IDS = Object.fromEntries(FIELD_NAMES.map((name, index) => [name, index]));
const HEADING_ICON =
  '<span class="heading-collapse-indicator collapse-indicator collapse-icon"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="svg-icon right-triangle"><path d="M3 8L12 17L21 8"></path></svg></span>';
const FOLDER_ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="svg-icon right-triangle"><path d="M3 8L12 17L21 8"></path></svg>';
const COLLAPSE_ALL_ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></svg>';

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) {
      continue;
    }

    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      parsed[token] = next;
      index += 1;
      continue;
    }

    parsed[token] = 'true';
  }

  return parsed;
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}

function readText(targetPath) {
  return fs.readFileSync(targetPath, 'utf8');
}

function writeText(targetPath, content) {
  ensureDir(path.dirname(targetPath));
  fs.writeFileSync(targetPath, content, 'utf8');
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function normalizePath(value) {
  return value.split(path.sep).join('/');
}

function slugify(value) {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-') || 'page';
}

function unique(array) {
  return [...new Set(array)];
}

function getPathToRoot(outputPath) {
  const directory = path.posix.dirname(outputPath);
  if (directory === '.') {
    return '.';
  }

  return directory
    .split('/')
    .map(() => '..')
    .join('/');
}

function getRootOutputPath(sourceRel) {
  switch (sourceRel) {
    case 'Home.md':
      return 'index.html';
    case 'Topics.md':
      return 'topics.html';
    case 'Updates.md':
      return 'updates.html';
    case 'Now.md':
      return 'now.html';
    case 'About.md':
      return 'about.html';
    case 'Garden-Structure.md':
      return 'garden-structure.html';
    case 'AGENTS.md':
      return 'agents.html';
    case 'index.md':
      return 'catalog.html';
    case 'log.md':
      return 'log.html';
    default:
      return `${slugify(path.posix.basename(sourceRel, '.md'))}.html`;
  }
}

function getOutputPath(sourceRel) {
  if (!sourceRel.includes('/')) {
    return getRootOutputPath(sourceRel);
  }

  const directory = path.posix.dirname(sourceRel);
  const basename = path.posix.basename(sourceRel, '.md');
  const filename = basename === 'index' ? 'index.html' : `${slugify(basename)}.html`;
  return path.posix.join(directory, filename);
}

function collectMarkdownFiles(basePath, folderName) {
  const absoluteFolder = path.join(basePath, folderName);
  if (!fs.existsSync(absoluteFolder)) {
    return [];
  }

  const collected = [];

  function walk(currentDir) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(absolutePath);
        continue;
      }

      if (!entry.isFile() || !entry.name.endsWith('.md')) {
        continue;
      }

      collected.push(normalizePath(path.relative(basePath, absolutePath)));
    }
  }

  walk(absoluteFolder);
  return collected;
}

function rootFileRank(sourceRel) {
  const index = ROOT_PAGE_ORDER.indexOf(sourceRel);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function folderRank(folderName, parentPath) {
  if (parentPath === '') {
    const index = ROOT_FOLDER_ORDER.indexOf(folderName);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  }

  if (parentPath === 'topics') {
    const index = TOPICS_FOLDER_ORDER.indexOf(folderName);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  }

  if (parentPath === 'topics/llm-platforms') {
    const index = LLM_SUBFOLDER_ORDER.indexOf(folderName);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  }

  return Number.MAX_SAFE_INTEGER;
}

function fileRank(note, parentPath) {
  if (parentPath === '') {
    return rootFileRank(note.sourceRel);
  }

  const basename = path.posix.basename(note.sourceRel);
  if (basename === 'index.md') {
    return -1;
  }

  return Number.MAX_SAFE_INTEGER;
}

function buildNoteRegistry(vaultPath) {
  const sourceFiles = [
    ...ROOT_PAGE_ORDER.filter((relativePath) => fs.existsSync(path.join(vaultPath, relativePath))),
    ...collectMarkdownFiles(vaultPath, 'topics'),
    ...collectMarkdownFiles(vaultPath, 'updates'),
    ...collectMarkdownFiles(vaultPath, 'raw')
  ];

  const registry = sourceFiles.map((sourceRel) => {
    const sourceAbs = path.join(vaultPath, sourceRel);
    const outputPath = getOutputPath(sourceRel);
    const noteName = path.posix.basename(sourceRel, '.md');
    const relativeKey = normalizePath(sourceRel.slice(0, -3));
    const showInTree = sourceRel.startsWith('topics/') || sourceRel.startsWith('updates/') || !sourceRel.includes('/');
    return {
      sourceRel,
      sourceAbs,
      outputPath,
      noteName,
      relativeKey,
      showInTree
    };
  });

  const basenameCounts = new Map();
  for (const note of registry) {
    basenameCounts.set(note.noteName, (basenameCounts.get(note.noteName) || 0) + 1);
  }

  const linkMap = new Map();
  for (const note of registry) {
    linkMap.set(note.relativeKey, note.outputPath);
    if ((basenameCounts.get(note.noteName) || 0) === 1 || note.sourceRel === 'index.md') {
      linkMap.set(note.noteName, note.outputPath);
    }
  }

  return { registry, linkMap };
}

function convertWikiLinks(markdown, linkMap) {
  return markdown.replace(/!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/gu, (match, rawTarget, rawLabel) => {
    if (match.startsWith('!')) {
      return rawLabel || path.posix.basename(rawTarget.trim());
    }

    const target = rawTarget.trim().replaceAll('\\', '/');
    const label = (rawLabel || path.posix.basename(target)).trim();
    const href = linkMap.get(target) || linkMap.get(path.posix.basename(target));
    if (!href) {
      return label;
    }

    return `[${label}](${href})`;
  });
}

function plainTextFromMarkdown(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[\[[^\]]+\]\]/g, ' ')
    .replace(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/gu, (_, target, label) => label || path.posix.basename(target.trim()))
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/^>\s?/gm, '')
    .replace(/<\/?[^>]+>/g, ' ')
    .replace(/\r/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function getMarkdownDescription(markdown) {
  const plainText = plainTextFromMarkdown(markdown);
  for (const line of plainText.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    return trimmed.length > 160 ? `${trimmed.slice(0, 160)}` : trimmed;
  }

  return '';
}

function renderInlineMarkdown(value) {
  const pattern = /`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)/g;
  let result = '';
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(value)) !== null) {
    result += escapeHtml(value.slice(lastIndex, match.index));
    if (match[1] !== undefined) {
      result += `<code>${escapeHtml(match[1])}</code>`;
    } else {
      result += `<a href="${escapeHtml(match[3])}">${escapeHtml(match[2])}</a>`;
    }

    lastIndex = match.index + match[0].length;
  }

  result += escapeHtml(value.slice(lastIndex));
  return result;
}

function renderMarkdownToHtml(markdown) {
  const lines = markdown.replace(/\r/g, '').split('\n');
  const htmlParts = [];
  let inCodeBlock = false;
  let codeLanguage = '';
  let codeLines = [];
  let paragraphLines = [];
  let listType = null;
  let listItems = [];

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return;
    }

    htmlParts.push(`<p>${renderInlineMarkdown(paragraphLines.join(' '))}</p>`);
    paragraphLines = [];
  }

  function flushList() {
    if (!listType) {
      return;
    }

    htmlParts.push(`<${listType}>${listItems.map((item) => `<li>${item}</li>`).join('')}</${listType}>`);
    listType = null;
    listItems = [];
  }

  function flushCodeBlock() {
    if (!inCodeBlock) {
      return;
    }

    const languageClass = codeLanguage ? ` class="language-${escapeHtml(codeLanguage)}"` : '';
    htmlParts.push(`<pre><code${languageClass}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    inCodeBlock = false;
    codeLanguage = '';
    codeLines = [];
  }

  for (const line of lines) {
    const codeFence = line.match(/^```(.*)$/);
    if (codeFence) {
      flushParagraph();
      flushList();
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLanguage = codeFence[1].trim();
        codeLines = [];
      } else {
        flushCodeBlock();
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      htmlParts.push(`<h${level} id="${slugify(text)}">${renderInlineMarkdown(text)}</h${level}>`);
      continue;
    }

    if (listType && listItems.length > 0 && /^\s{2,}\S/.test(line)) {
      listItems[listItems.length - 1] += `<br>${renderInlineMarkdown(line.trim())}`;
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*+]\s+(.*)$/);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== 'ul') {
        flushList();
      }

      listType = 'ul';
      listItems.push(renderInlineMarkdown(unorderedMatch[1].trim()));
      continue;
    }

    const orderedMatch = line.match(/^\s*\d+\.\s+(.*)$/);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== 'ol') {
        flushList();
      }

      listType = 'ol';
      listItems.push(renderInlineMarkdown(orderedMatch[1].trim()));
      continue;
    }

    flushList();
    paragraphLines.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushCodeBlock();

  return htmlParts.join('');
}

function buildContentConversion(markdown) {
  const rawHtml = renderMarkdownToHtml(markdown);
  const headings = [];
  let headingIndex = 0;
  const bodyHtml = rawHtml.replace(/<h([1-6]) id="[^"]*">(.*?)<\/h\1>/gms, (_, level, innerHtml) => {
    headingIndex += 1;
    const headingText = innerHtml.replace(/<[^>]+>/g, '').trim();
    const headingId = `heading-${headingIndex}`;
    headings.push({
      heading: headingText,
      level: Number(level),
      id: headingId
    });

    const safeHeading = escapeHtml(headingText);
    return `<div class="el-h${level}"><h${level} data-heading="${safeHeading}" dir="auto" class="heading" id="${headingId}">${HEADING_ICON}${innerHtml}</h${level}></div>`;
  });

  const links = unique(
    [...bodyHtml.matchAll(/<a[^>]+href="([^"]+)"/g)]
      .map((match) => match[1])
      .filter((href) => href && !href.startsWith('#') && !href.startsWith('?') && !href.startsWith('http'))
  );

  return { bodyHtml, headings, links };
}

function buildOutlineHtml(outputPath, headings) {
  const items = headings
    .map((heading) => {
      const safeText = escapeHtml(heading.heading);
      return `<div class="tree-item" data-depth="${heading.level}"><a class="tree-item-self is-clickable" href="${outputPath}#${heading.id}" data-path="#${heading.id}"><div class="tree-item-inner heading-link" heading-name="${safeText}">${safeText}</div></a><div class="tree-item-children"></div></div>`;
    })
    .join('');

  return `<div id="outline" class=" tree-container"><div class="feature-header"><div class="feature-title">Table Of Contents</div><button class="clickable-icon nav-action-button tree-collapse-all" aria-label="Collapse All">${COLLAPSE_ALL_ICON}</button></div>${items}</div>`;
}

function buildContentBlock(title, bodyHtml) {
  const safeTitle = escapeHtml(title);
  return `<div class="markdown-preview-sizer markdown-preview-section"><div class="header"><h1 class="page-title heading inline-title" id="page-title">${safeTitle}</h1><div class="data-bar"></div></div><div class="markdown-preview-pusher" style="width: 1px; height: 0.1px; margin-bottom: 0px;"></div>${bodyHtml}<div class="footer"><div class="data-bar"></div></div></div></div></div>`;
}

function applyPageTemplate(template, { outputPath, title, description, contentBlock, outlineHtml, baseHref }) {
  const safeTitle = escapeHtml(title);
  const safeDescription = escapeHtml(description);
  const safeOutputPath = escapeHtml(outputPath);

  let page = template
    .replace(/<title>.*?<\/title>/s, `<title>${safeTitle}</title>`)
    .replace(/<base href="[^"]*">/, `<base href="${baseHref}">`)
    .replace(/<meta name="pathname" content="[^"]*">/, `<meta name="pathname" content="${safeOutputPath}">`)
    .replace(/<meta name="description" content="[^"]*">/, `<meta name="description" content="${safeDescription}">`)
    .replace(/<meta property="og:title" content="[^"]*">/, `<meta property="og:title" content="${safeTitle}">`)
    .replace(/<meta property="og:description" content="[^"]*">/, `<meta property="og:description" content="${safeDescription}">`)
    .replace(/<meta property="og:url" content="[^"]*">/, `<meta property="og:url" content="${safeOutputPath}">`);

  const centerStartMarker = '<div class="markdown-preview-sizer markdown-preview-section">';
  const rightContentMarker = '<div id="right-content" class="leaf"';
  const centerStart = page.indexOf(centerStartMarker);
  const rightContentStart = page.indexOf(rightContentMarker);
  if (centerStart === -1 || rightContentStart === -1) {
    throw new Error('Template center content markers were not found.');
  }

  page = `${page.slice(0, centerStart)}${contentBlock}${page.slice(rightContentStart)}`;

  const outlineStartMarker = '<div id="outline" class=" tree-container">';
  const outlineEndMarker = '</div></div></div><script defer="">let rs';
  const outlineStart = page.indexOf(outlineStartMarker);
  const outlineEnd = page.indexOf(outlineEndMarker, outlineStart);
  if (outlineStart === -1 || outlineEnd === -1) {
    throw new Error('Template outline markers were not found.');
  }

  page = `${page.slice(0, outlineStart)}${outlineHtml}${page.slice(outlineEnd)}`;
  return page;
}

function buildVisibleTree(notes) {
  const root = { path: '', folders: new Map(), files: [] };

  for (const note of notes.filter((entry) => entry.showInTree)) {
    const parts = note.sourceRel.split('/');
    if (parts.length === 1) {
      root.files.push(note);
      continue;
    }

    let current = root;
    let currentPath = '';
    for (const segment of parts.slice(0, -1)) {
      const nextPath = currentPath ? `${currentPath}/${segment}` : segment;
      if (!current.folders.has(segment)) {
        current.folders.set(segment, { path: nextPath, folders: new Map(), files: [] });
      }

      current = current.folders.get(segment);
      currentPath = nextPath;
    }

    current.files.push(note);
  }

  return root;
}

function sortTree(node) {
  node.files.sort((left, right) => {
    const leftRank = fileRank(left, node.path);
    const rightRank = fileRank(right, node.path);
    if (leftRank !== rightRank) {
      return leftRank - rightRank;
    }

    return left.outputPath.localeCompare(right.outputPath, 'ko');
  });

  const sortedFolders = [...node.folders.entries()].sort(([leftName], [rightName]) => {
    const leftRank = folderRank(leftName, node.path);
    const rightRank = folderRank(rightName, node.path);
    if (leftRank !== rightRank) {
      return leftRank - rightRank;
    }

    return leftName.localeCompare(rightName, 'ko');
  });

  node.folders = new Map(sortedFolders);
  for (const child of node.folders.values()) {
    sortTree(child);
  }
}

function renderTreeNode(node, depth, orderedVisibleNotes) {
  const fileMarkup = node.files
    .map((note) => {
      orderedVisibleNotes.push(note);
      const label = escapeHtml(path.posix.basename(note.sourceRel, '.md'));
      const safePath = escapeHtml(note.sourceRel);
      return `<div class="tree-item is-collapsed nav-file" data-depth="${depth}"><a class="tree-item-self is-clickable nav-file-title" href="${note.outputPath}" data-path="${safePath}"><div class="tree-item-inner nav-file-title-content">${label}</div></a><div class="tree-item-children nav-file-children"></div></div>`;
    })
    .join('');

  const folderMarkup = [...node.folders.values()]
    .map((folder) => {
      const children = renderTreeNode(folder, depth + 1, orderedVisibleNotes);
      const label = escapeHtml(path.posix.basename(folder.path));
      const safePath = escapeHtml(folder.path);
      return `<div class="tree-item mod-collapsible is-collapsed nav-folder" data-depth="${depth}"><div class="tree-item-self is-clickable mod-collapsible nav-folder-title" data-path="${safePath}"><div class="tree-item-icon collapse-icon is-collapsed nav-folder-collapse-indicator">${FOLDER_ICON}</div><div class="tree-item-inner nav-folder-title-content">${label}</div></div><div class="tree-item-children nav-folder-children" style="display: none;">${children}</div></div>`;
    })
    .join('');

  return `${fileMarkup}${folderMarkup}`;
}

function buildFileTreeHtml(notes) {
  const tree = buildVisibleTree(notes);
  sortTree(tree);
  const orderedVisibleNotes = [];
  const content = renderTreeNode(tree, 1, orderedVisibleNotes);
  const html = `<div id="file-explorer" class=" tree-container"><div class="feature-header"><div class="feature-title">github.vault</div><button class="clickable-icon nav-action-button tree-collapse-all is-collapsed" aria-label="Collapse All">${COLLAPSE_ALL_ICON}</button></div>${content}</div>`;
  return { html, orderedVisibleNotes };
}

function tokenize(text) {
  if (!text) {
    return [];
  }

  return text
    .toLowerCase()
    .split(/[\n\r\p{Z}\p{P}]+/u)
    .map((token) => token.trim())
    .filter(Boolean);
}

function buildSearchIndex(noteEntries) {
  const tokenIndex = new Map();
  const documentIds = {};
  const storedFields = {};
  const fieldLength = {};
  const averageFieldLength = new Array(FIELD_NAMES.length).fill(0);

  noteEntries.forEach((note, offset) => {
    const documentId = String(offset + 1);
    documentIds[documentId] = note.outputPath;

    storedFields[documentId] = {
      title: note.title,
      aliases: note.aliases,
      headers: note.headers.map((heading) => heading.heading),
      tags: note.tags,
      path: note.outputPath
    };

    const fieldTokens = {
      title: tokenize(note.title),
      aliases: note.aliases.flatMap((alias) => tokenize(alias)),
      headers: note.headers.flatMap((heading) => tokenize(heading.heading)),
      tags: note.tags.flatMap((tag) => tokenize(tag)),
      path: tokenize(note.outputPath),
      content: tokenize(note.searchContent)
    };

    fieldLength[documentId] = FIELD_NAMES.map((fieldName) => fieldTokens[fieldName].length);
    FIELD_NAMES.forEach((fieldName, fieldIndex) => {
      averageFieldLength[fieldIndex] += fieldTokens[fieldName].length;
      const frequency = new Map();
      for (const token of fieldTokens[fieldName]) {
        frequency.set(token, (frequency.get(token) || 0) + 1);
      }

      for (const [token, count] of frequency.entries()) {
        if (!tokenIndex.has(token)) {
          tokenIndex.set(token, {});
        }

        const posting = tokenIndex.get(token);
        const fieldId = FIELD_IDS[fieldName];
        if (!posting[fieldId]) {
          posting[fieldId] = {};
        }

        posting[fieldId][documentId] = count;
      }
    });
  });

  if (noteEntries.length > 0) {
    for (let index = 0; index < averageFieldLength.length; index += 1) {
      averageFieldLength[index] /= noteEntries.length;
    }
  }

  const index = [...tokenIndex.entries()]
    .sort(([left], [right]) => left.localeCompare(right, 'ko'))
    .map(([token, posting]) => [token, posting]);

  return {
    documentCount: noteEntries.length,
    nextId: noteEntries.length + 1,
    documentIds,
    fieldIds: FIELD_IDS,
    fieldLength,
    averageFieldLength,
    storedFields,
    dirtCount: noteEntries.length,
    index,
    serializationVersion: 2
  };
}

function buildRedirectHtml(targetPath) {
  const escapedTarget = escapeHtml(targetPath);
  return `<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta http-equiv="refresh" content="0; url=${escapedTarget}"><title>Redirect</title><link rel="canonical" href="${escapedTarget}"></head><body><p><a href="${escapedTarget}">이동</a></p><script>location.replace(${JSON.stringify(targetPath)});</script></body></html>`;
}

function cleanGeneratedOutputs(exportPath) {
  const directoriesToRemove = ['topics', 'updates', 'raw', 'wiki'];
  for (const relativePath of directoriesToRemove) {
    const absolutePath = path.join(exportPath, relativePath);
    if (fs.existsSync(absolutePath)) {
      fs.rmSync(absolutePath, { recursive: true, force: true });
    }
  }

  const filesToRemove = [
    'index.html',
    'topics.html',
    'updates.html',
    'now.html',
    'about.html',
    'garden-structure.html',
    'catalog.html',
    'agents.html',
    'log.html'
  ];

  for (const relativePath of filesToRemove) {
    const absolutePath = path.join(exportPath, relativePath);
    if (fs.existsSync(absolutePath)) {
      fs.rmSync(absolutePath, { force: true });
    }
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const vaultPath = args['--vault-path'] || DEFAULT_VAULT_PATH;
  const exportPath = args['--export-path'] || DEFAULT_EXPORT_PATH;

  const metadataPath = path.join(exportPath, 'site-lib', 'metadata.json');
  if (!fs.existsSync(metadataPath)) {
    throw new Error(`Metadata file not found: ${metadataPath}`);
  }

  const existingMetadata = JSON.parse(readText(metadataPath));
  const templateCandidates = [
    path.join(SCRIPT_DIR, 'template-shell.html'),
    path.join(exportPath, 'index.html'),
    path.join(exportPath, 'wiki', 'home.html')
  ];
  const templatePath = templateCandidates.find((candidate) => fs.existsSync(candidate));
  if (!templatePath) {
    throw new Error(`Template page not found. Checked: ${templateCandidates.join(', ')}`);
  }

  const template = readText(templatePath);
  const { registry, linkMap } = buildNoteRegistry(vaultPath);

  cleanGeneratedOutputs(exportPath);

  const builtNotes = [];
  for (const note of registry) {
    const originalMarkdown = readText(note.sourceAbs);
    const linkedMarkdown = convertWikiLinks(originalMarkdown, linkMap);
    const { bodyHtml, headings, links } = buildContentConversion(linkedMarkdown);
    const description = getMarkdownDescription(originalMarkdown);
    const title = headings.find((heading) => heading.level === 1)?.heading || note.noteName;
    const baseHref = getPathToRoot(note.outputPath);
    const outlineHtml = buildOutlineHtml(note.outputPath, headings);
    const contentBlock = buildContentBlock(title, bodyHtml);
    const html = applyPageTemplate(template, {
      outputPath: note.outputPath,
      title,
      description,
      contentBlock,
      outlineHtml,
      baseHref
    });

    const outputAbs = path.join(exportPath, note.outputPath);
    writeText(outputAbs, html);

    const fileStats = fs.statSync(note.sourceAbs);
    builtNotes.push({
      ...note,
      title,
      description,
      headings,
      links,
      aliases: [],
      tags: [],
      author: '',
      coverImageURL: '',
      pathToRoot: baseHref,
      createdTime: Math.round(fileStats.birthtimeMs || fileStats.ctimeMs || Date.now()),
      modifiedTime: Math.round(fileStats.mtimeMs || Date.now()),
      sourceSize: fileStats.size,
      type: 'markdown',
      searchContent: plainTextFromMarkdown(originalMarkdown)
    });
  }

  const { html: fileTreeHtml, orderedVisibleNotes } = buildFileTreeHtml(builtNotes);
  writeText(path.join(exportPath, 'site-lib', 'html', 'file-tree-content.html'), fileTreeHtml);

  const visibleOrder = new Map(orderedVisibleNotes.map((note, index) => [note.outputPath, index + 1]));
  const noteInfoByPath = new Map();
  for (const note of builtNotes) {
    noteInfoByPath.set(note.outputPath, {
      title: note.title,
      icon: '',
      description: note.description,
      aliases: note.aliases,
      tags: note.tags,
      inlineTags: [],
      frontmatterTags: [],
      headers: note.headings,
      links: note.links.filter((href) => builtNotes.some((candidate) => candidate.outputPath === href)),
      author: note.author,
      coverImageURL: note.coverImageURL,
      fullURL: note.outputPath,
      pathToRoot: note.pathToRoot,
      attachments: [],
      createdTime: note.createdTime,
      modifiedTime: note.modifiedTime,
      sourceSize: note.sourceSize,
      sourcePath: note.sourceRel,
      exportPath: note.outputPath,
      showInTree: note.showInTree,
      treeOrder: visibleOrder.get(note.outputPath) || 0,
      backlinks: [],
      type: note.type,
      searchContent: note.searchContent
    });
  }

  for (const noteInfo of noteInfoByPath.values()) {
    for (const target of noteInfo.links) {
      const targetInfo = noteInfoByPath.get(target);
      if (targetInfo) {
        targetInfo.backlinks.push(noteInfo.fullURL);
      }
    }
  }

  for (const noteInfo of noteInfoByPath.values()) {
    noteInfo.backlinks = unique(noteInfo.backlinks).sort((left, right) => left.localeCompare(right, 'ko'));
  }

  const preservedFileInfo = Object.fromEntries(
    Object.entries(existingMetadata.fileInfo || {}).filter(([filePath]) => filePath.startsWith('site-lib/'))
  );
  const preservedAssets = unique(
    [
      ...(existingMetadata.attachments || []),
      ...(existingMetadata.allFiles || [])
    ].filter((filePath) => filePath.startsWith('site-lib/'))
  );

  const webpages = {};
  const fileInfo = { ...preservedFileInfo };
  for (const [outputPath, noteInfo] of noteInfoByPath.entries()) {
    webpages[outputPath] = { ...noteInfo };
    fileInfo[outputPath] = {
      createdTime: noteInfo.createdTime,
      modifiedTime: noteInfo.modifiedTime,
      sourceSize: noteInfo.sourceSize,
      sourcePath: noteInfo.sourcePath,
      exportPath: noteInfo.exportPath,
      showInTree: noteInfo.showInTree,
      treeOrder: noteInfo.treeOrder,
      backlinks: noteInfo.backlinks,
      type: noteInfo.type,
      data: null
    };
    delete webpages[outputPath].searchContent;
  }

  const noteOutputPaths = builtNotes.map((note) => note.outputPath);
  const shownInTree = orderedVisibleNotes.map((note) => note.outputPath);
  const sourceToTarget = Object.fromEntries(builtNotes.map((note) => [note.sourceRel, note.outputPath]));

  const updatedMetadata = {
    ...existingMetadata,
    shownInTree,
    attachments: preservedAssets,
    allFiles: unique([...noteOutputPaths, ...preservedAssets]),
    webpages,
    fileInfo,
    sourceToTarget,
    modifiedTime: Date.now(),
    siteName: 'github.vault',
    vaultName: 'github.vault'
  };

  writeText(metadataPath, JSON.stringify(updatedMetadata));

  const searchIndex = buildSearchIndex([...noteInfoByPath.values()]);
  writeText(path.join(exportPath, 'site-lib', 'search-index.json'), JSON.stringify(searchIndex));

  const redirects = [
    ['wiki/home.html', '../index.html'],
    ['wiki/사용-방법.html', '../topics/notes-and-methods/사용-방법.html'],
    ['wiki/llm-maintained-wiki-pattern.html', '../topics/notes-and-methods/llm-maintained-wiki-pattern.html'],
    ['wiki/source-karpathy-llm-wiki.html', '../topics/notes-and-methods/source-karpathy-llm-wiki.html']
  ];

  for (const [redirectPath, targetPath] of redirects) {
    writeText(path.join(exportPath, redirectPath), buildRedirectHtml(targetPath));
  }

  console.log(`Generated ${builtNotes.length} pages into ${exportPath}`);
}

main();
