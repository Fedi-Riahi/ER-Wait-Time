/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0a0a0f',
          800: '#12121a',
          700: '#1a1a2a',
          600: '#1e1e2e',
        },
        accent: {
          DEFAULT: '#3b82f6',
          hover:   '#2563eb',
        }
      }
    }
  },
  plugins: []
}
