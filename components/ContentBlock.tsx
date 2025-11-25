"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ContentBlockProps {
    color: "pink" | "blue" | "purple" | "gray";
    title: string;
    children: React.ReactNode;
    className?: string;
}

export default function ContentBlock({ color, title, children, className }: ContentBlockProps) {
    const colorVariants = {
        pink: "bg-anime-pink-light border-anime-pink/20",
        blue: "bg-anime-blue-light border-anime-blue/20",
        purple: "bg-anime-purple-light border-anime-purple/20",
        gray: "bg-anime-gray-light border-anime-gray/20",
    };

    const titleColorVariants = {
        pink: "text-anime-pink",
        blue: "text-cyan-500",
        purple: "text-anime-purple",
        gray: "text-slate-500",
    };

    const particleColorVariants = {
        pink: "bg-anime-pink",
        blue: "bg-anime-blue",
        purple: "bg-anime-purple",
        gray: "bg-anime-gray",
    };

    // Generate random particles
    const particles = Array.from({ length: 25 }).map((_, i) => ({
        id: i,
        size: Math.random() * 20 + 5, // 5px to 25px
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        duration: Math.random() * 20 + 10, // 10s to 30s
        delay: Math.random() * 5,
    }));

    return (
        <section className={cn("w-full py-20 px-6 border-y relative overflow-hidden", colorVariants[color], className)}>
            {/* Floating Particles Background */}
            <div className="absolute inset-0 w-full h-full pointer-events-none overflow-hidden">
                {particles.map((particle) => (
                    <motion.div
                        key={particle.id}
                        className={cn("absolute rounded-full opacity-10", particleColorVariants[color])}
                        style={{
                            width: particle.size,
                            height: particle.size,
                            left: particle.left,
                            top: particle.top,
                        }}
                        animate={{
                            y: [0, -100, 0],
                            x: [0, 50, 0],
                            opacity: [0.1, 0.3, 0.1],
                        }}
                        transition={{
                            duration: particle.duration,
                            repeat: Infinity,
                            ease: "easeInOut",
                            delay: particle.delay,
                        }}
                    />
                ))}
            </div>

            <div className="max-w-6xl mx-auto relative z-10">
                <motion.h2
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.6 }}
                    className={cn("text-3xl md:text-4xl font-bold mb-8 text-center", titleColorVariants[color])}
                >
                    {title}
                </motion.h2>
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                    className="bg-white/60 backdrop-blur-sm rounded-3xl p-8 shadow-sm border border-white/50"
                >
                    {children}
                </motion.div>
            </div>
        </section>
    );
}
