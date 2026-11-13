// @ts-check
// The lucem documentation: Starlight (Astro), in the neuro-kitchen family (the same structure as
// dsp-kitchen's docs) with lucem's own look (src/styles/theme.css).
// Served at https://neuro-kitchen.github.io/lucem/ (GitHub Pages of this repository).
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightSidebarTopics from 'starlight-sidebar-topics';

const repo = 'https://github.com/neuro-kitchen/lucem';

/**
 * An entry point: its own sidebar, opened at the folder's index page. `items` lists the pages
 * in order; by default, every page of the folder.
 */
const topic = (label, directory, items = [{ autogenerate: { directory } }]) => ({
  label,
  link: `/${directory}/`,
  items,
});

/** A chapter of a topic: a few pages, folded unless the current page is in it. */
const chapter = (label, directory, collapsed = false) => ({ label, collapsed, items: [{ autogenerate: { directory, collapsed } }] });

export default defineConfig({
  site: 'https://neuro-kitchen.github.io',
  base: '/lucem',

  // Fonts as files, not data: URIs, so font-src 'self' covers them.
  vite: { build: { assetsInlineLimit: 0 } },

  // The Content-Security-Policy is added after the build by scripts/csp.mjs (Astro's own option
  // misses Starlight's head script and style attributes).

  integrations: [
    starlight({
      title: 'lucem',
      description: 'GPU image analysis for histology and microscopy: an app, a command line and headless pipelines.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: repo }],
      editLink: { baseUrl: `${repo}/edit/main/docs/` },
      lastUpdated: true,
      expressiveCode: { defaultProps: { wrap: true } },
      routeMiddleware: './src/route-middleware.ts',
      customCss: ['@fontsource-variable/ibm-plex-sans', '@fontsource/ibm-plex-mono/400.css', './src/styles/theme.css'],
      // One entry point per purpose, each with its own sidebar. The topics are tabs in the header
      // (src/components/Header.astro, grouped by src/data/nav.mjs).
      components: {
        Header: './src/components/Header.astro',
        Sidebar: './src/components/Sidebar.astro',
      },
      plugins: [
        starlightSidebarTopics([
          topic('Learn', 'learn', ['learn', chapter('Get started', 'learn/get-started'), chapter('Concepts', 'learn/concepts')]),
          topic('App', 'app'),
          topic('Histology', 'histology'),
          topic('Reference', 'reference', ['reference', 'reference/operations', chapter('Command line', 'reference/cli'), chapter('Files', 'reference/files'), chapter('Rust API', 'reference/rust', true)]),
          topic('Models', 'models'),
          topic('Extend', 'extend'),
          topic('Deploy', 'deploy'),
          topic('Changelog', 'changelog'),
        ]),
      ],
    }),
  ],
});
