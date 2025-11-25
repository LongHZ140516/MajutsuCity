"use client";

import Link from "next/link";
import Image from "next/image"
import { Github, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface HeroProps {
    className?: string;
    backgroundColor?: string; // e.g., "bg-anime-pink-light/30" or hex code
    patternColor?: string;    // e.g., "text-anime-pink" or hex code (via style if needed, but class is easier)
}

const authors = [
    { name: "Zilong Huang", link: "https://serein-six.vercel.app/", avatar: "avatar/zilonghuang.jpg", initial: "ZH", superscript: "1,*" },
    { name: "Jun He", link: "https://github.com/Hejun0915", avatar: "avatar/junhe.jpg", initial: "JH", superscript: "1,*" },
    { name: "Xiaobin Huang", link: "https://yejy53.github.io/", avatar: "avatar/xiaobinhuang.jpg", initial: "XH", superscript: "1" },
    { name: "Ziyi Xiong", link: "#", avatar: "avatar/ziyixiong.jpg", initial: "ZX", superscript: "1" },
    { name: "Yang Luo", link: "https://scholar.google.com.hk/citations?user=FAY1XUMAAAAJ&hl=zh-CN", avatar: "avatar/yangluo.jpg", initial: "YL", superscript: "1" },
    { name: "Junyan Ye", link: "https://yejy53.github.io/", avatar: "avatar/junyanye.jpg", initial: "JY", superscript: "1" },
    { name: "Weijia Li", link: "https://liweijia.github.io/", avatar: "avatar/weijiali.jpg", initial: "WL", superscript: "1" },
    { name: "Yiping Chen", link: "https://scholar.google.com/citations?user=e9lv2fUAAAAJ&hl=en", avatar: "avatar/yipingchen.jpg", initial: "YC", superscript: "1,†" },
    { name: "Ting Han", link: "https://ting-devin-han.github.io/Homepage-main/", avatar: "/avatar/tinghan.jpg", initial: "TH", superscript: "1,†" },
];

export default function Hero({
    className,
    backgroundColor = "bg-anime-pink-light/30",
    patternColor = "anime-pink"
}: HeroProps) {
    return (
        <section className={cn("relative w-full min-h-screen flex flex-col items-center justify-center overflow-hidden", backgroundColor, className)}>
            {/* Background Pattern */}
            <div className={cn("absolute inset-0 z-0 pointer-events-none", patternColor)}>
                <div className="absolute inset-0 bg-ghost-pattern opacity-40"></div>
                <div className="absolute inset-0 bg-gradient-to-t from-white via-transparent to-transparent"></div>
            </div>

            {/* Content */}
            <div className="relative z-10 flex flex-col items-center text-center px-4 max-w-4xl mx-auto space-y-8 mt-20">
                {/* Title */}
                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="font-extrabold text-slate-800 tracking-tight"
                >
                    {/* <span className="block text-anime-pink mb-2 text-5xl">MajutsuCity:</span> */}
                    <div className="flex flex-col items-center">
                        <Image
                            src="images/logo2.png"
                            alt="logo"
                            width={0}
                            height={0}
                            sizes="20vw"
                            className="w-[20vw] h-auto object-contain"
                        />
                    </div>
                    <span className="block text-5xl">Language-driven Aesthetic-adaptive City Generation with Controllable 3D Assets and Layouts</span>
                </motion.h1>

                {/* Authors & Institutions */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                    className="flex flex-col items-center gap-2 text-lg md:text-xl text-slate-600"
                >
                    <div className="flex flex-wrap justify-center gap-4 font-medium">
                        <TooltipProvider>
                            {authors.map((author, index) => (
                                <Tooltip key={index}>
                                    <TooltipTrigger asChild>
                                        <Link href={author.link} className="hover:text-anime-pink transition-colors">
                                            {author.name} <sup>{author.superscript}</sup>
                                        </Link>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <div className="flex items-center gap-2">
                                            <Avatar>
                                                <AvatarImage src={author.avatar} alt={author.name} />
                                                <AvatarFallback>{author.initial}</AvatarFallback>
                                            </Avatar>
                                            {/* <p>{author.name}</p> */}
                                        </div>
                                    </TooltipContent>
                                </Tooltip>
                            ))}
                        </TooltipProvider>
                    </div>
                    <div className="text-slate-500">
                        1 Sun Yat-sen University
                    </div>
                    <div className="text-slate-500">
                        * Equal Contribution,   † Corresponding authors
                    </div>
                </motion.div>

                {/* Buttons */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
                    className="flex flex-wrap justify-center gap-4 mt-8"
                >
                    <Link
                        href="#"
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-anime-pink text-white font-bold shadow-lg hover:bg-anime-pink/90 hover:scale-105 transition-all duration-200"
                    >
                        <FileText className="w-5 h-5" />
                        <span>Arxiv</span>
                    </Link>
                    <Link
                        href="#"
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-slate-700 font-bold shadow-lg border border-slate-200 hover:bg-slate-50 hover:scale-105 transition-all duration-200"
                    >
                        <Github className="w-5 h-5" />
                        <span>Code (coming soon)</span>
                    </Link>
                </motion.div>
            </div>
        </section>
    );
}
