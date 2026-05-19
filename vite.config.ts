import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves this repo at /sec-financial-chatbot/
export default defineConfig({
  plugins: [react()],
  base: '/sec-financial-chatbot/',
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
