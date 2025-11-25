"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Citation() {
    const [copied, setCopied] = useState(false);

    const bibtex = `@article{majutsu2024city,
  title={MajutsuCity: Language-driven Aesthetic-adaptive City Generation with Controllable 3D Assets and Layouts},
  author={Huang, Zilong and He, Jun and Huang, Xiaobin and Xiong, Ziyi and Luo, Yang and Ye, Junyan and Li, Weijia and Chen, Yiping and Han, Ting},
  journal={arXiv preprint arXiv:24xx.xxxxx},
  year={2024}
}`;

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(bibtex);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error("Failed to copy text: ", err);
        }
    };

    return (
        <section className="w-full py-12 px-6 bg-white border-t border-slate-100">
            <div className="max-w-6xl mx-auto">
                <h2 className="text-2xl font-bold text-slate-800 mb-6">Citation</h2>
                <div className="relative bg-slate-50 rounded-xl overflow-hidden border border-slate-200">
                    <div className="absolute top-4 right-4">
                        <button
                            onClick={handleCopy}
                            className={cn(
                                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200 border shadow-sm",
                                copied
                                    ? "bg-green-50 border-green-200 text-green-600"
                                    : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                            )}
                        >
                            {copied ? (
                                <>
                                    <Check className="w-4 h-4" />
                                    <span>Copied!</span>
                                </>
                            ) : (
                                <>
                                    <Copy className="w-4 h-4" />
                                    <span>Copy BibTeX</span>
                                </>
                            )}
                        </button>
                    </div>
                    <pre className="p-6 pt-14 overflow-x-auto text-sm md:text-base font-mono text-slate-600 leading-relaxed">
                        <code>{bibtex}</code>
                    </pre>
                </div>
            </div>
        </section>
    );
}
