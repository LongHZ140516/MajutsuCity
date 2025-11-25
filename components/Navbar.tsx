import Link from "next/link";
import { Github, Home, FileText } from "lucide-react";

export default function Navbar() {
    return (
        <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-white/80 backdrop-blur-md border-b border-white/20 transition-all duration-300">
            <div className="flex items-center gap-4">
                {/* Logo/Name */}
                <Link href="/" className="text-2xl font-bold text-anime-pink hover:opacity-80 transition-opacity">
                    MajutsuCity
                </Link>
            </div>

            <div className="flex items-center gap-6">
                {/* Icons/Links */}
                <Link href="/" className="text-anime-pink hover:text-anime-pink/70 transition-colors">
                    <Home className="w-6 h-6" />
                </Link>
                <Link href="/" target="_blank" className="text-anime-pink hover:text-anime-pink/70 transition-colors">
                    <FileText className="w-6 h-6" />
                </Link>
                <Link href="https://github.com/LongHZ140516/MajutsuCity" target="_blank" className="text-anime-pink hover:text-anime-pink/70 transition-colors">
                    <Github className="w-6 h-6" />
                </Link>
            </div>
        </nav>
    );
}
