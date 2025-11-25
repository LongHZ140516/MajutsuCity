import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Image from "next/image"
import ContentBlock from "@/components/ContentBlock";
import Citation from "@/components/Citation";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col font-sans">
      <Navbar />

      <Hero />

      {/* Abstract - Pink */}
      <ContentBlock color="pink" title="Abstract">
        <div className="prose prose-lg max-w-none text-slate-700 space-y-6">
          <p className="text-lg leading-relaxed">
            Generating realistic 3D cities is fundamental to world models, virtual reality, and game development, where an ideal urban scene must satisfy both stylistic diversity, fine-grained, and controllability. However, existing methods struggle to balance the creative flexibility offered by text-based generation with the object-level editability enabled by explicit structural representations.
          </p>
          <p className="text-lg leading-relaxed">
            We introduce <strong className="text-anime-blue">MajutsuCity</strong>, a natural language–driven and aesthetically adaptive framework for synthesizing structurally consistent and stylistically diverse 3D urban scenes. MajutsuCity represents a city as a composition of controllable layouts, assets, and materials, and operates through a four-stage pipeline. To extend controllability beyond initial generation, we further integrate <strong className="text-anime-pink">MajutsuAgent</strong>, an interactive language-grounded editing agent that supports five object-level operations.
          </p>
          <p className="text-lg leading-relaxed">
            To support photorealistic and customizable scene synthesis, we also construct <strong className="text-anime-purple">MajutsuDataset</strong>, a high-quality multimodal dataset containing 2D semantic layouts and height maps, diverse 3D building assets, and curated PBR materials and skyboxes, each accompanied by detailed annotations.
          </p>
          <p className="text-lg leading-relaxed">
            Meanwhile, we develop <strong>a practical set of evaluation metrics</strong>, covering key dimensions such as structural consistency, scene complexity, material fidelity, and lighting atmosphere.
            Extensive experiments demonstrate MajutsuCity reduces layout FID by <i>83.7%</i> compared with CityDreamer and by <i>20.1%</i> over CityCraft. Our method ranks first across all AQS and RDR scores, outperforming existing methods by a clear margin. These results confirm MajutsuCity as a new state-of-the-art in geometric fidelity, stylistic adaptability, and semantic controllability for 3D city generation. We expect our framework can inspire new avenues of research in 3D city generation.
          </p>

          <div className="flex flex-col items-center gap-4 mt-8">
            <div className="relative w-full">
              <Image
                src="/images/flag.jpg"
                alt="MajutsuCity Overview"
                width={0}
                height={0}
                sizes="100vw"
                className="w-full h-auto object-contain border-2 border-dashed border-anime-pink/30 rounded-xl"
              />
            </div>
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Figure 1: <strong>MajutsuCity</strong> is a language–driven, aesthetic-adaptive system that unifies controllable urban scene generation and interactive editing within a single framework. Conditioned on textual instructions, the framework synthesizes a complete stylized city through layout–height creation, asset instantiation, and terrain/material generation, and further enables iterative refinement through five atomic editing operations. This paradigm forms the core contribution of MajutsuCity, empowering users to create and continuously modify large-scale, stylistically diverse urban scenes through natural language.
            </p>
          </div>
        </div>
      </ContentBlock>

      {/* Method - Blue */}
      <ContentBlock color="blue" title="Method">
        <div className="space-y-8">
          <div className="prose prose-lg max-w-none text-slate-700">
            <p className="text-lg leading-relaxed">
              To enable controllable generation of object-level urban scenes directly from natural language, we propose MajutsuCity framework that bridges high-level textual intent and structured 3D scene composition.
            </p>
          </div>

          <div className="flex flex-col items-center gap-4">
            <div className="relative w-full">
              <Image
                src="/images/pipeline.jpg"
                alt="Methodology Pipeline"
                width={0}
                height={0}
                sizes="100vw"
                className="w-full h-auto object-contain border-2 border-dashed border-anime-blue/30 rounded-xl"
              />
            </div>
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Figure 2: <strong>Overview of the proposed MajutsuCity framework</strong>. MajutsuCity is an aesthetic-adaptive generative framework that enables controllable, object-level 3D urban scene generation from natural language descriptions. It consists of Scene Design, Layout Generation, Assets & Materials Generation, and Scene Generation.
            </p>
          </div>

          <div className="prose prose-lg max-w-none text-slate-700">
            <p className="text-lg leading-relaxed">
              As shown in <i>Figure. 2</i>, the framework has <strong>four</strong> major stages:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
            <div className="bg-white/50 rounded-xl p-6 border border-anime-blue/20 shadow-sm">
              <h3 className="text-xl font-bold text-anime-blue mb-2">Scene Design</h3>
              <p className="text-slate-600">Converting textual requirements into structured and consistent design guidance.</p>
            </div>
            <div className="bg-white/50 rounded-xl p-6 border border-anime-blue/20 shadow-sm">
              <h3 className="text-xl font-bold text-anime-blue mb-2">Layout Generation</h3>
              <p className="text-slate-600">Synthesizing spatially coherent urban layouts and height maps under semantic and topological constraints.</p>
            </div>
            <div className="bg-white/50 rounded-xl p-6 border border-anime-blue/20 shadow-sm">
              <h3 className="text-xl font-bold text-anime-blue mb-2">Assets & Materials Generation</h3>
              <p className="text-slate-600">Producing high-fidelity building-level 3D assets and material maps and skybox.</p>
            </div>
            <div className="bg-white/50 rounded-xl p-6 border border-anime-blue/20 shadow-sm">
              <h3 className="text-xl font-bold text-anime-blue mb-2">Scene Generation</h3>
              <p className="text-slate-600">Composing assets and environmental layers into a coherent and renderable 3D city.</p>
            </div>
          </div>

          <p className="text-lg leading-relaxed">
            Furthermore, we develop <strong>MajutsuAgent</strong>, an interactive editing agent enables fine-grained, human-in-the-loop scene manipulation with high controllability and consistency.
          </p>
        </div>
      </ContentBlock>

      {/* Dataset - Purple */}
      <ContentBlock color="purple" title="Dataset">
        <div className="space-y-8">
          <div className="prose prose-lg max-w-none text-slate-700">
            <p className="text-lg leading-relaxed">
              To enhance the realism and controllability of 3D scene generation, we introduce <strong className="text-anime-purple">MajutsuDataset</strong>, a high-quality multimodal dataset designed for text-guided 3D scene synthesis. As illustrated in <i>Figure. 3</i>, the dataset comprises <strong>three</strong> major components: <strong>Layout/Elevation</strong>, <strong>3D Building Models</strong>, and <strong>Material Assets</strong>.
            </p>
          </div>

          <div className="flex flex-col items-center gap-4">
            <div className="relative w-[60vw]">
              <Image
                src="/images/dataset.png"
                alt="MajutsuDataset"
                width={0}
                height={0}
                sizes="60vw"
                className="w-[60vw] h-auto object-contain border-2 border-dashed border-anime-purple/30 rounded-xl"
              />
            </div>
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Figure 3: <strong>Overview of MajutsuDataset</strong>, a high-quality multimodal dataset designed for text-guided 3D urban scene generation. (a) The OSM-based Layout/Elevation subset provides paired semantic layout maps, height maps, and detailed textual descriptions generated by GPT-5-mini. (b) The 3D Building Models subset includes 1,000 assets covering diverse architectural styles. (c) The Texture Map subset contains a large-scale library of seamlessly tilable PBR materials and HDR skybox maps.
            </p>
          </div>
        </div>
      </ContentBlock>

      {/* Results - Gray */}
      <ContentBlock color="gray" title="Results">
        <div className="space-y-8">
          {/* <div className="prose prose-lg max-w-none text-slate-700">
            <p className="text-lg leading-relaxed">
              Extensive experiments demonstrate MajutsuCity reduces layout FID by 83.7% compared with CityDreamer and by 20.1% over CityCraft. Our method ranks first across all AQS and RDR scores, outperforming existing methods by a clear margin.
            </p>
          </div> */}
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Table 1: <strong>Quantitative comparison of Layout Generation.</strong>
            </p>
            <div className="relative w-full pb-1 flex justify-center">
              <Image
                src="/images/quantitative_layout.jpg"
                alt="layout table"
                width={0}
                height={0}
                sizes="100vw"
                className="w-[25vw] h-auto object-contain border-2 border-dashed border-slate-300 rounded-xl"
              />
            </div>
          </div>
          <div className="flex flex-col items-center gap-4">
            <div className="relative w-full pb-1 flex justify-center">
              <Image
                src="/images/layout.jpg"
                alt="Layout Result"
                width={0}
                height={0}
                sizes="100vw"
                className="w-full h-auto object-contain border-2 border-dashed border-slate-300 rounded-xl"
              />
            </div>
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Figure 4: <strong>Qualitative comparison of city layouts generation.</strong> Our method yields more realistic and coherent urban layouts than prior InfiniteGAN, CityDreamer and CityCraft.
            </p>
          </div>
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Table 2: <strong>Absolute Quantitative Scoring (AQS) and Relative Dimension Ranking (RDR) for city scene generation.</strong> For each metric, we report both GPT-based and user-based scores.
            </p>
            <div className="relative w-full pb-1 flex justify-center">
              <Image
                src="/images/quantitative_scene.jpg"
                alt="Scene table"
                width={0}
                height={0}
                sizes="100vw"
                className="w-[70vw] h-auto object-contain border-2 border-dashed border-slate-300 rounded-xl"
              />
            </div>
          </div>
          <div className="flex flex-col items-center gap-4 mb-4">
            <div className="relative w-full">
              <Image
                src="/images/scene.jpg"
                alt="scene Result"
                width={0}
                height={0}
                sizes="100vw"
                className="w-full h-auto object-contain border-2 border-dashed border-slate-300 rounded-xl"
              />
            </div>
            <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
              Figure 5: <strong>Qualitative comparison of city scene generation.</strong> We compare our method with CityDreamer, GaussianCity, UrbanWorld and CityCraft across two representative scenes. Our approach produces scenes with higher geometric fidelity, better multi-view consistency, and richer stylistic diversity than all baselines.
            </p>
          </div>
        </div>
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-[full]">
            <Image
              src="/images/four_styles.jpg"
              alt="4 styles Result"
              width={0}
              height={0}
              sizes="50vw"
              className="w-[50vw] h-auto object-contain border-2 border-dashed border-slate-300 rounded-xl"
            />
          </div>
          <p className="text-sm text-slate-500 font-medium italic text-center items-centers">
            Figure 6: <strong>Style-driven city generation results.</strong> Four city scenes
            with different well-known styles generated by MajutsuCity show
            high fidelity and strong intra-style consistency.
          </p>
        </div>

      </ContentBlock>

      <Citation />

      <footer className="py-8 text-center text-slate-400 text-sm bg-white border-t border-slate-100">
        <p>© 2025 Majutsu City. All rights reserved.</p>
      </footer>
    </main>
  );
}

