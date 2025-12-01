import Header from '@/components/Header';
import EmotionDetector from '@/components/EmotionDetector';
import Dashboard from '@/components/Dashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      <Header />
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-slate-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
            Real-time Emotion Analysis
          </h1>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-slate-500">
            Detect facial emotions in real-time using deep learning directly in your browser.
          </p>
        </div>

        <Tabs defaultValue="analysis" className="w-full">
          <div className="flex justify-center mb-8">
            <TabsList className="grid w-full max-w-md grid-cols-2">
              <TabsTrigger value="analysis">Live Analysis</TabsTrigger>
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="analysis">
            <EmotionDetector />
          </TabsContent>

          <TabsContent value="dashboard">
            <Dashboard />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
