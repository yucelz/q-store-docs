// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
  integrations: [
      starlight({
          title: 'Q-Store',
          description: 'Q-Store Database Architecture Documentation',
          logo: {
              src: './src/assets/logo.svg',
          },
          social: [
              { icon: 'github', label: 'GitHub', href: 'https://github.com/yucelz/q-store' }
          ],
          sidebar: [
              {
                  label: 'Getting Started',
                  items: [
                      { label: 'Introduction', slug: 'index' },
                      { label: 'Quick Start', slug: 'getting-started/quick-start' },
                      { label: 'Installation', slug: 'getting-started/installation' },
                      { label: 'Q-Store v3.2', slug: 'getting-started/version-3-2' },
                  ],
              },
              {
                  label: 'Core Concepts',
                  items: [
                      { label: 'Architecture Overview', slug: 'concepts/architecture' },
                      { label: 'Quantum Principles', slug: 'concepts/quantum-principles' },
                      { label: 'Hybrid Design', slug: 'concepts/hybrid-design' },
                  ],
              },
              {
                  label: 'System Components',
                  items: [
                      { label: 'State Manager', slug: 'components/state-manager' },
                      { label: 'Entanglement Registry', slug: 'components/entanglement-registry' },
                      { label: 'Quantum Circuit Builder', slug: 'components/circuit-builder' },
                      { label: 'Tunneling Engine', slug: 'components/tunneling-engine' },
                      { label: 'Classical Backend', slug: 'components/classical-backend' },
                  ],
              },
              {
                  label: 'IonQ Integration',
                  items: [
                      { label: 'Overview', slug: 'ionq/overview' },
                      { label: 'SDK Integration', slug: 'ionq/sdk-integration' },
                  ],
              },
              {
                  label: 'Domain Applications',
                  items: [
                      { label: 'Financial Services', slug: 'applications/financial' },
                      { label: 'ML Model Training', slug: 'applications/ml-training' },
                      { label: 'Recommendation Systems', slug: 'applications/recommendations' },
                      { label: 'Scientific Computing', slug: 'applications/scientific' },
                  ],
              },


              {
                  label: 'Advanced Topics',
                  items: [
                      { label: 'Database Performance', slug: 'advanced/performance' },
                      { label: 'ML Training Performance', slug: 'advanced/ml-training-performance' },
                  ],
              },
              {
                  label: 'Project Info',
                  items: [
                      { label: 'License', slug: 'license' },
                      { label: 'Contact', slug: 'contact' },
                  ],
              },
          ],
      }),
	],

  adapter: vercel(),
});