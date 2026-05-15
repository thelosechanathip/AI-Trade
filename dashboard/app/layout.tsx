import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-Trade | Live Dashboard",
  description: "Professional automated trading system dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#050a14] text-gray-100 font-sans antialiased overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
