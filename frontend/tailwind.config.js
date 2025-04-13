/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
      extend: {
        animation: {
          'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        },
        keyframes: {
          'pulse-glow': {
            '0%, 100%': { opacity: 0.6 },
            '50%': { opacity: 1 },
          },
        },
      },
    },
    plugins: [],
  }