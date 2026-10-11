import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import {
  UploadCloud,
  Camera,
  RotateCcw,
  Download,
  Code,
  SlidersHorizontal,
  Layers,
  Info,
  Zap,
  FolderOpen
} from 'lucide-react'

export default function App() {
  const [task, setTask] = useState('detect')
  const [conf, setConf] = useState(25)
  const [iou, setIou] = useState(45)
  const [device, setDevice] = useState('auto')
  const [activeView, setActiveView] = useState('annotated')
  const [isLoading, setIsLoading] = useState(false)
  const [samples, setSamples] = useState([])
  const [systemInfo, setSystemInfo] = useState(null)
  const [selectedCategories, setSelectedCategories] = useState(new Set())
  const [detectionData, setDetectionData] = useState(null)
  const [currentFile, setCurrentFile] = useState(null)
  const [currentSampleName, setCurrentSampleName] = useState(null)
  const [highlightedId, setHighlightedId] = useState(null)

  // Webcam State
  const [webcamOpen, setWebcamOpen] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const fileInputRef = useRef(null)

  // Fetch System Info & Samples on Mount
  useEffect(() => {
    fetchInfo()
    fetchSamples()
  }, [])

  const fetchInfo = async () => {
    try {
      const res = await fetch('/api/info')
      const data = await res.json()
      setSystemInfo(data)
      if (data.model?.capabilities?.categories) {
        setSelectedCategories(new Set(Object.keys(data.model.capabilities.categories)))
      }
    } catch (err) {
      console.error('Error fetching info:', err)
    }
  }

  const fetchSamples = async () => {
    try {
      const res = await fetch('/api/samples')
      const data = await res.json()
      setSamples(data)
    } catch (err) {
      console.error('Error fetching samples:', err)
    }
  }

  const runDetection = async (overrideFile = null, overrideSample = null) => {
    const fileToUse = overrideFile !== null ? overrideFile : currentFile
    const sampleToUse = overrideSample !== null ? overrideSample : currentSampleName

    if (!fileToUse && !sampleToUse) return

    setIsLoading(true)
    try {
      const formData = new FormData()
      if (fileToUse) {
        formData.append('file', fileToUse)
      } else if (sampleToUse) {
        formData.append('sample_name', sampleToUse)
      }

      formData.append('task', task)
      formData.append('conf_threshold', (conf / 100).toString())
      formData.append('iou_threshold', (iou / 100).toString())
      formData.append('device', device)

      // Filter classes if some categories deselected
      const allCats = systemInfo?.model?.capabilities?.categories || {}
      if (selectedCategories.size < Object.keys(allCats).length) {
        const allowed = []
        selectedCategories.forEach(catName => {
          if (allCats[catName]?.classes) {
            allowed.push(...allCats[catName].classes)
          }
        })
        formData.append('classes_filter', allowed.join(','))
      }

      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      })
      if (!res.ok) throw new Error('Detection failed')
      const data = await res.json()
      setDetectionData(data)
    } catch (err) {
      alert('Detection Error: ' + err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setCurrentFile(file)
      setCurrentSampleName(null)
      runDetection(file, null)
    }
  }

  const handleSelectSample = (sampleFile) => {
    setCurrentFile(null)
    setCurrentSampleName(sampleFile)
    runDetection(null, sampleFile)
  }

  const handleReset = () => {
    setCurrentFile(null)
    setCurrentSampleName(null)
    setDetectionData(null)
  }

  const toggleCategory = (catName) => {
    const next = new Set(selectedCategories)
    if (next.has(catName)) {
      if (next.size > 1) next.delete(catName)
    } else {
      next.add(catName)
    }
    setSelectedCategories(next)
  }

  // Webcam
  const startWebcam = async () => {
    setWebcamOpen(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      alert('Cannot access camera: ' + err.message)
      setWebcamOpen(false)
    }
  }

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    setWebcamOpen(false)
  }

  const captureWebcam = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      const file = new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' })
      setCurrentFile(file)
      setCurrentSampleName(null)
      stopWebcam()
      runDetection(file, null)
    }, 'image/jpeg', 0.95)
  }

  const downloadAnnotatedImage = () => {
    if (!detectionData?.annotated_image) return
    const a = document.createElement('a')
    a.href = detectionData.annotated_image
    a.download = `yolov8x_${Date.now()}.jpg`
    a.click()
  }

  const exportJSON = () => {
    if (!detectionData) return
    const blob = new Blob([JSON.stringify(detectionData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `yolov8x_detections_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const categories = systemInfo?.model?.capabilities?.categories || {}

  const taskOptions = [
    { id: 'detect', name: 'Object Detection', desc: 'Bounding boxes (yolov8x.pt)', icon: 'crop_free' },
    { id: 'segment', name: 'Instance Segmentation', desc: 'Pixel masks (yolov8x-seg)', icon: 'polyline' },
    { id: 'pose', name: 'Pose Estimation', desc: '17 keypoints (yolov8x-pose)', icon: 'accessibility_new' },
    { id: 'classify', name: 'Classification', desc: 'Top ImageNet (yolov8x-cls)', icon: 'category' },
    { id: 'obb', name: 'Oriented Bounding Box', desc: 'Rotated boxes (yolov8x-obb)', icon: 'rotate_90_degrees_ccw' }
  ]

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-900 select-none">
      {/* Top Header */}
      <header className="h-14 bg-white border-b border-slate-200 px-5 flex items-center justify-end z-20 shadow-xs gap-3">
        {/* Capabilities Dialog */}
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs text-slate-700">
              <span className="material-symbols-outlined text-base">dataset</span>
              YOLOv8x Capabilities
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-base text-slate-900">
                <span className="material-symbols-outlined text-black">model_training</span>
                YOLOv8x Specifications &amp; Architecture
              </DialogTitle>
            </DialogHeader>

            <div className="space-y-4 text-xs text-slate-600">
              <div>
                <h4 className="font-semibold text-slate-900 mb-1 text-sm">Model Overview</h4>
                <p className="leading-relaxed">
                  <strong>YOLOv8x</strong> is the largest and most accurate deep learning model in the YOLOv8 family developed by Ultralytics.
                  It contains ~68.2 Million parameters and achieves state-of-the-art <strong>53.9 mAP</strong> on COCO (50–95).
                </p>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-sm">Supported 80 COCO Classes (12 Categories)</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {Object.entries(categories).map(([catName, catData]) => (
                    <div key={catName} className="p-2.5 rounded-lg border border-slate-200 bg-slate-50">
                      <div className="flex items-center gap-1.5 font-semibold text-xs mb-1.5" style={{ color: catData.color }}>
                        <span className="material-symbols-outlined text-sm">{catData.icon}</span>
                        {catName} ({catData.classes.length})
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {catData.classes.map(c => (
                          <span key={c} className="bg-white border border-slate-200 px-1.5 py-0.5 rounded text-[11px] text-slate-600">
                            {c}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold text-slate-900 mb-1.5 text-sm">17 Human Pose Keypoints</h4>
                <div className="flex flex-wrap gap-1.5">
                  {(systemInfo?.model?.capabilities?.pose_keypoints || []).map((kp, idx) => (
                    <Badge key={kp} variant="outline" className="text-[11px]">
                      {idx + 1}. {kp}
                    </Badge>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <h4 className="font-semibold text-slate-900 mb-1.5 text-sm">Fine-Tuning &amp; Use Cases</h4>
                <div className="grid grid-cols-2 gap-1.5 text-slate-600">
                  <div>&bull; Manufacturing defect inspection</div>
                  <div>&bull; License plates &amp; vehicles</div>
                  <div>&bull; PPE helmets &amp; safety</div>
                  <div>&bull; Facial &amp; retail tracking</div>
                  <div>&bull; Autonomous robotics</div>
                  <div>&bull; Agricultural pests &amp; weeds</div>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Webcam Button */}
        <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs text-slate-700" onClick={startWebcam}>
          <Camera className="w-3.5 h-3.5" />
          Webcam
        </Button>

        {/* Reset Button */}
        <Button variant="ghost" size="sm" className="h-8 gap-1.5 text-xs text-slate-600" onClick={handleReset}>
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </Button>
      </header>

      {/* Main Body Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT SIDEBAR: Controls & Tabs */}
        <aside className="w-88 bg-white border-r border-slate-200 flex flex-col z-10">
          <Tabs defaultValue="inference" className="flex flex-col h-full">
            <div className="p-3 border-b border-slate-200 bg-slate-50/70">
              <TabsList className="grid w-full grid-cols-3 bg-slate-200/70">
                <TabsTrigger value="inference" className="gap-1.5 text-xs font-medium">
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  Inference
                </TabsTrigger>
                <TabsTrigger value="classes" className="gap-1.5 text-xs font-medium">
                  <Layers className="w-3.5 h-3.5" />
                  Classes
                </TabsTrigger>
                <TabsTrigger value="specs" className="gap-1.5 text-xs font-medium">
                  <Info className="w-3.5 h-3.5" />
                  Specs
                </TabsTrigger>
              </TabsList>
            </div>

            {/* TAB 1: INFERENCE */}
            <TabsContent value="inference" className="flex-1 overflow-y-auto p-4 space-y-4 m-0">
              {/* Task Mode */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold text-slate-800">
                  <span>Vision Task</span>
                  <Badge variant="default" className="text-[11px] font-semibold capitalize bg-black text-white">
                    {task}
                  </Badge>
                </div>
                <div className="space-y-1.5">
                  {taskOptions.map(t => (
                    <button
                      key={t.id}
                      onClick={() => setTask(t.id)}
                      className={`w-full p-2.5 rounded-lg border text-left flex items-center gap-3 transition-all cursor-pointer ${
                        task === t.id
                          ? 'border-black bg-slate-100/70 shadow-xs'
                          : 'border-slate-200 bg-white hover:bg-slate-50'
                      }`}
                    >
                      <div className={`w-8 h-8 rounded flex items-center justify-center shrink-0 ${
                        task === t.id ? 'bg-black text-white' : 'bg-slate-100 text-slate-600'
                      }`}>
                        <span className="material-symbols-outlined text-lg">{t.icon}</span>
                      </div>
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-slate-900">{t.name}</div>
                        <div className="text-[11px] text-slate-500 truncate">{t.desc}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Confidence Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold text-slate-800">
                  <span>Confidence Threshold</span>
                  <span className="font-mono text-black text-xs font-bold">{conf}%</span>
                </div>
                <Slider
                  value={[conf]}
                  onValueChange={(val) => setConf(val[0])}
                  min={5}
                  max={95}
                  step={1}
                />
                <span className="text-[11px] text-slate-400">Discard predictions with lower certainty</span>
              </div>

              {/* IoU Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-semibold text-slate-800">
                  <span>IoU / NMS Threshold</span>
                  <span className="font-mono text-black text-xs font-bold">{iou}%</span>
                </div>
                <Slider
                  value={[iou]}
                  onValueChange={(val) => setIou(val[0])}
                  min={10}
                  max={90}
                  step={5}
                />
                <span className="text-[11px] text-slate-400">Non-maximum suppression overlap filter</span>
              </div>

              <Separator />

              {/* Compute Hardware */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-800">Compute Hardware</span>
                <div className="grid grid-cols-3 gap-1.5">
                  {['auto', 'cuda', 'cpu'].map(dev => (
                    <Button
                      key={dev}
                      size="sm"
                      variant={device === dev ? 'default' : 'outline'}
                      className="text-xs capitalize h-8 font-medium"
                      onClick={() => setDevice(dev)}
                    >
                      {dev}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Run Inference Action */}
              <Button
                variant="default"
                className="w-full mt-2 gap-2 h-9 text-xs font-semibold shadow-sm cursor-pointer"
                onClick={() => {
                  if (currentFile || currentSampleName) runDetection()
                  else fileInputRef.current?.click()
                }}
              >
                <Zap className="w-4 h-4 fill-current" />
                Run YOLOv8x Inference
              </Button>
            </TabsContent>

            {/* TAB 2: CLASSES */}
            <TabsContent value="classes" className="flex-1 overflow-y-auto p-4 space-y-3 m-0">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-800">Filter COCO Categories</span>
                <div className="flex gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => setSelectedCategories(new Set(Object.keys(categories)))}
                  >
                    All
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => setSelectedCategories(new Set([Object.keys(categories)[0]]))}
                  >
                    Reset
                  </Button>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5 pt-1">
                {Object.entries(categories).map(([catName, catData]) => {
                  const active = selectedCategories.has(catName)
                  return (
                    <button
                      key={catName}
                      onClick={() => toggleCategory(catName)}
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-all cursor-pointer ${
                        active
                          ? 'bg-black text-white border-black'
                          : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: catData.color }} />
                      <span>{catName}</span>
                      <span className="text-[10px] opacity-70">({catData.classes.length})</span>
                    </button>
                  )
                })}
              </div>

              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs space-y-1">
                <div className="font-semibold text-slate-800">
                  {selectedCategories.size === Object.keys(categories).length
                    ? 'All 80 COCO Classes Active'
                    : `${selectedCategories.size} of 12 Categories Selected`}
                </div>
                <p className="text-slate-500 text-[11px] leading-relaxed">
                  Predictions will automatically filter to the selected categories during inference.
                </p>
              </div>
            </TabsContent>

            {/* TAB 3: SPECS */}
            <TabsContent value="specs" className="flex-1 overflow-y-auto p-4 space-y-3 m-0">
              <Card>
                <CardHeader className="p-3 pb-2">
                  <CardTitle className="text-xs">Model Specifications</CardTitle>
                </CardHeader>
                <CardContent className="p-3 pt-0 space-y-2 text-xs">
                  <div className="flex justify-between"><span className="text-slate-500">Model</span><span className="font-semibold">YOLOv8x</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Parameters</span><span className="font-semibold">68.2 Million</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">COCO mAP (50-95)</span><span className="font-semibold text-black">53.9 mAP</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Resolution</span><span className="font-semibold">640 &times; 640 px</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Weights File</span><span className="font-semibold">yolov8x.pt (136.9 MB)</span></div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="p-3 pb-2">
                  <CardTitle className="text-xs">Hardware Acceleration</CardTitle>
                </CardHeader>
                <CardContent className="p-3 pt-0 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Device</span>
                    <span className="font-semibold">{systemInfo?.hardware?.device_name || 'Standard CPU'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">CUDA GPU</span>
                    <span className="font-semibold text-emerald-600">
                      {systemInfo?.hardware?.cuda_available ? 'Available' : 'CPU Mode'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </aside>

        {/* CENTER WORKSPACE */}
        <main className="flex-1 flex flex-col bg-slate-50/50 overflow-hidden relative">
          {/* Workspace Toolbar */}
          <div className="h-14 bg-white border-b border-slate-200 px-5 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Button
                variant="default"
                size="sm"
                className="gap-2 h-8 text-xs font-semibold cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadCloud className="w-4 h-4" />
                Upload Image
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileUpload}
              />

              <Separator orientation="vertical" className="h-5" />

              {/* Sample Pills */}
              <div className="flex items-center gap-1.5 overflow-x-auto">
                <span className="text-xs text-slate-400 font-medium whitespace-nowrap">Samples:</span>
                {samples.map(s => (
                  <Button
                    key={s.id}
                    variant="outline"
                    size="sm"
                    className="h-7 px-2.5 text-xs text-slate-600 hover:text-black"
                    onClick={() => handleSelectSample(s.file)}
                  >
                    {s.name.split(' ')[0]}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex bg-slate-100 p-0.5 rounded-lg border border-slate-200">
                <button
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-all cursor-pointer ${
                    activeView === 'annotated' ? 'bg-black text-white shadow-xs' : 'text-slate-500 hover:text-black'
                  }`}
                  onClick={() => setActiveView('annotated')}
                >
                  Annotated
                </button>
                <button
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-all cursor-pointer ${
                    activeView === 'original' ? 'bg-black text-white shadow-xs' : 'text-slate-500 hover:text-black'
                  }`}
                  onClick={() => setActiveView('original')}
                >
                  Original
                </button>
              </div>

              <Button
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs text-slate-700"
                onClick={downloadAnnotatedImage}
                disabled={!detectionData}
              >
                <Download className="w-3.5 h-3.5" />
                Save Image
              </Button>
            </div>
          </div>

          {/* Canvas Viewport */}
          <div className="flex-1 relative overflow-auto flex items-center justify-center p-6 bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] bg-[size:18px_18px]">
            {/* Loading Indicator */}
            {isLoading && (
              <div className="absolute inset-0 bg-white/80 backdrop-blur-xs flex flex-col items-center justify-center gap-3 z-30">
                <div className="w-10 h-10 border-3 border-slate-300 border-t-black rounded-full animate-spin" />
                <span className="text-sm font-semibold text-slate-900">YOLOv8x Analyzing Image...</span>
              </div>
            )}

            {/* Empty Dropzone State */}
            {!detectionData && (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  const file = e.dataTransfer.files?.[0]
                  if (file) {
                    setCurrentFile(file)
                    setCurrentSampleName(null)
                    runDetection(file, null)
                  }
                }}
                className="max-w-md w-full bg-white border-2 border-dashed border-slate-300 rounded-2xl p-8 flex flex-col items-center text-center gap-4 shadow-sm hover:border-black hover:bg-slate-50/50 transition-all cursor-pointer"
              >
                <div className="w-14 h-14 rounded-full bg-slate-100 text-slate-900 flex items-center justify-center">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-slate-900">Drop an image here</h3>
                  <p className="text-xs text-slate-500 mt-1">Supports PNG, JPG, JPEG, WEBP</p>
                </div>

                <Button variant="default" size="sm" className="gap-2 text-xs">
                  <FolderOpen className="w-4 h-4" />
                  Browse Files
                </Button>

                <div className="w-full pt-4 border-t border-slate-100">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2.5">
                    Or select a preset test sample:
                  </span>
                  <div className="flex justify-center gap-2 flex-wrap" onClick={(e) => e.stopPropagation()}>
                    {samples.map(s => (
                      <button
                        key={s.id}
                        onClick={() => handleSelectSample(s.file)}
                        className="w-16 h-12 rounded-lg overflow-hidden border-2 border-transparent hover:border-black transition-all shadow-xs cursor-pointer"
                      >
                        <img src={s.url} alt={s.name} className="w-full h-full object-cover" />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Visualizer Display */}
            {detectionData && (
              <div className="relative max-w-full max-h-full rounded-xl shadow-lg overflow-hidden bg-white border border-slate-200">
                <div className="relative inline-block leading-none">
                  <img
                    src={activeView === 'annotated' ? detectionData.annotated_image : detectionData.original_image}
                    alt="YOLOv8x Visualizer"
                    className="max-w-full max-h-[calc(100vh-160px)] object-contain block"
                  />

                  {/* Interactive Box Overlay */}
                  <div className="absolute inset-0 pointer-events-none">
                    {detectionData.detections.map(det => {
                      const [nx, ny, nw, nh] = det.bbox_normalized
                      const isHigh = highlightedId === det.id
                      return (
                        <div
                          key={det.id}
                          className="absolute pointer-events-auto cursor-pointer transition-all rounded-xs"
                          style={{
                            left: `${nx * 100}%`,
                            top: `${ny * 100}%`,
                            width: `${nw * 100}%`,
                            height: `${nh * 100}%`,
                            border: isHigh ? '2px solid #000000' : `2px dashed ${det.color}`,
                            backgroundColor: isHigh ? 'rgba(0, 0, 0, 0.15)' : 'transparent',
                            boxShadow: isHigh ? '0 0 10px rgba(0, 0, 0, 0.5)' : 'none'
                          }}
                          onMouseEnter={() => setHighlightedId(det.id)}
                          onMouseLeave={() => setHighlightedId(null)}
                          title={`${det.class_name} (${det.confidence}%)`}
                        />
                      )
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* RIGHT SIDEBAR: Detection Inspector */}
        <aside className="w-96 bg-white border-l border-slate-200 flex flex-col z-10">
          <div className="p-3.5 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-black text-lg">radar</span>
              <span className="font-semibold text-xs text-slate-900">Detection Results</span>
            </div>
            <Badge variant="secondary" className="text-xs font-semibold">
              {detectionData?.metrics?.total_detections || 0} Found
            </Badge>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Top Metrics Cards */}
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-center">
                <div className="text-base font-bold text-slate-900">
                  {detectionData?.metrics?.total_detections || 0}
                </div>
                <div className="text-[10px] uppercase font-semibold text-slate-400">Objects</div>
              </div>
              <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-center">
                <div className="text-base font-bold text-slate-900">
                  {detectionData?.metrics?.inference_time_ms || 0}<span className="text-[10px] font-normal">ms</span>
                </div>
                <div className="text-[10px] uppercase font-semibold text-slate-400">Latency</div>
              </div>
              <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-center">
                <div className="text-base font-bold text-slate-900">
                  {detectionData?.metrics?.fps || 0}
                </div>
                <div className="text-[10px] uppercase font-semibold text-slate-400">FPS</div>
              </div>
            </div>

            {/* Category summary badges */}
            {detectionData?.category_counts && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(detectionData.category_counts).map(([cat, count]) => {
                  const color = categories[cat]?.color || '#64748b'
                  return (
                    <Badge
                      key={cat}
                      variant="outline"
                      className="gap-1 text-xs py-0.5"
                      style={{ color, borderColor: `${color}40`, backgroundColor: `${color}08` }}
                    >
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                      <span>{cat}: <strong>{count}</strong></span>
                    </Badge>
                  )
                })}
              </div>
            )}

            {/* Classification ranking if classify task */}
            {detectionData?.classification_top?.length > 0 && (
              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                  Top ImageNet Classes:
                </span>
                {detectionData.classification_top.map(item => (
                  <div key={item.rank} className="flex justify-between p-2 rounded border border-slate-200 bg-slate-50 text-xs">
                    <span className="font-semibold text-slate-800">{item.rank}. {item.class_name}</span>
                    <span className="font-mono text-black font-bold">{item.confidence}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Entities List */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Detected Objects
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-6 px-2 text-[10px] gap-1"
                onClick={exportJSON}
                disabled={!detectionData}
              >
                <Code className="w-3 h-3" />
                Export JSON
              </Button>
            </div>

            <div className="space-y-2">
              {!detectionData?.detections?.length ? (
                <div className="text-center py-10 text-slate-400 text-xs">
                  <span className="material-symbols-outlined text-3xl text-slate-300 block mb-1">image_search</span>
                  No detections to display yet.
                </div>
              ) : (
                detectionData.detections.map(det => {
                  const isHigh = highlightedId === det.id
                  return (
                    <Card
                      key={det.id}
                      className={`transition-all cursor-pointer ${
                        isHigh ? 'border-black bg-slate-50/80 shadow-xs' : 'hover:border-slate-300'
                      }`}
                      onMouseEnter={() => setHighlightedId(det.id)}
                      onMouseLeave={() => setHighlightedId(null)}
                    >
                      <CardContent className="p-2.5 flex items-center gap-3">
                        <div className="w-12 h-12 rounded-md bg-slate-100 overflow-hidden border border-slate-200 shrink-0 flex items-center justify-center">
                          {det.crop_base64 ? (
                            <img src={det.crop_base64} alt={det.class_name} className="w-full h-full object-cover" />
                          ) : (
                            <span className="material-symbols-outlined text-slate-400" style={{ color: det.color }}>
                              {det.icon}
                            </span>
                          )}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-center mb-0.5">
                            <span className="font-semibold text-xs text-slate-900 capitalize truncate">
                              {det.class_name}
                            </span>
                            <span className="text-xs font-bold text-black font-mono">
                              {det.confidence}%
                            </span>
                          </div>

                          <div className="flex items-center gap-2 text-[11px] text-slate-500 mb-1.5">
                            <span className="px-1.5 py-0.2 rounded font-medium text-[10px]" style={{ color: det.color, backgroundColor: `${det.color}15` }}>
                              {det.category}
                            </span>
                            <span>{det.width}&times;{det.height}px</span>
                          </div>

                          <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all" style={{ width: `${det.confidence}%`, backgroundColor: det.color }} />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Webcam Modal */}
      {webcamOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl max-w-md w-full border border-slate-200 shadow-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm text-slate-900 flex items-center gap-2">
                <Camera className="w-4 h-4 text-black" />
                Live Webcam Capture
              </h3>
              <button onClick={stopWebcam} className="text-slate-400 hover:text-slate-600 text-sm font-bold">
                ✕
              </button>
            </div>

            <video ref={videoRef} autoPlay playsInline className="w-full rounded-lg bg-black aspect-video object-cover" />
            <canvas ref={canvasRef} className="hidden" />

            <Button variant="default" className="w-full text-xs font-semibold h-9 gap-2" onClick={captureWebcam}>
              <Camera className="w-4 h-4" />
              Capture &amp; Analyze
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
