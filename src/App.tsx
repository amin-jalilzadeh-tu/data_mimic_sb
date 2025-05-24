import { useState, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Upload, Play, Database } from 'lucide-react'
import { useStore } from '@/lib/store'
import { DataTable } from '@/components/DataTable'
import Papa from 'papaparse'
import type { ColumnDef } from '@tanstack/react-table'

export default function App() {
  const {
    rawBuildingsFile,
    rawTimeSeriesFile,
    processedBuildingsFile,
    processedTimeSeriesFile,
    buildingsData,
    timeSeriesData,
    assignmentsData,
    progress,
    status,
    setRawBuildingsFile,
    setRawTimeSeriesFile,
    setProcessedBuildingsFile,
    setProcessedTimeSeriesFile,
    setBuildingsData,
    setTimeSeriesData,
    setAssignmentsData,
    setProgress,
    setStatus,
  } = useStore()

  const [buildingsColumns, setBuildingsColumns] = useState<ColumnDef<any, any>[]>([])
  const [timeSeriesColumns, setTimeSeriesColumns] = useState<ColumnDef<any, any>[]>([])
  const [assignmentsColumns, setAssignmentsColumns] = useState<ColumnDef<any, any>[]>([])

  const handleFileUpload = useCallback(async (file: File, type: string) => {
    const reader = new FileReader()
    reader.onload = async (e) => {
      const text = e.target?.result as string
      Papa.parse(text, {
        header: true,
        complete: (results) => {
          const columns: ColumnDef<any, any>[] = Object.keys(results.data[0] || {}).map(key => ({
            accessorKey: key,
            header: key,
          }))

          switch (type) {
            case 'rawBuildings':
              setRawBuildingsFile(file)
              setBuildingsColumns(columns)
              setBuildingsData(results.data)
              break
            case 'rawTimeSeries':
              setRawTimeSeriesFile(file)
              setTimeSeriesColumns(columns)
              setTimeSeriesData(results.data)
              break
            case 'processedBuildings':
              setProcessedBuildingsFile(file)
              setBuildingsColumns(columns)
              setBuildingsData(results.data)
              break
            case 'processedTimeSeries':
              setProcessedTimeSeriesFile(file)
              setTimeSeriesColumns(columns)
              setTimeSeriesData(results.data)
              break
          }
        }
      })
    }
    reader.readAsText(file)
  }, [])

  const runStep = async (step: number) => {
    setStatus(`Running Step ${step}...`)
    setProgress(0)
    
    // Simulate processing with progress updates
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 10
      })
    }, 500)

    // TODO: Add actual Python processing integration
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container py-4">
          <h1 className="text-2xl font-bold text-gray-900">Energy Community Data Pipeline</h1>
          {status && <p className="text-sm text-gray-600 mt-1">{status}</p>}
        </div>
      </header>

      <main className="container py-8">
        <div className="grid grid-cols-12 gap-8">
          {/* Sidebar */}
          <div className="col-span-4 space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Data Upload</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Raw Buildings CSV</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'rawBuildings')}
                    className="hidden"
                    id="rawBuildingsUpload"
                  />
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => document.getElementById('rawBuildingsUpload')?.click()}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    {rawBuildingsFile ? rawBuildingsFile.name : 'Upload File'}
                  </Button>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Raw Time Series CSV</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'rawTimeSeries')}
                    className="hidden"
                    id="rawTimeSeriesUpload"
                  />
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => document.getElementById('rawTimeSeriesUpload')?.click()}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    {rawTimeSeriesFile ? rawTimeSeriesFile.name : 'Upload File'}
                  </Button>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Processed Buildings CSV (Optional)</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'processedBuildings')}
                    className="hidden"
                    id="processedBuildingsUpload"
                  />
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => document.getElementById('processedBuildingsUpload')?.click()}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    {processedBuildingsFile ? processedBuildingsFile.name : 'Upload File'}
                  </Button>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Processed Time Series CSV (Optional)</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'processedTimeSeries')}
                    className="hidden"
                    id="processedTimeSeriesUpload"
                  />
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => document.getElementById('processedTimeSeriesUpload')?.click()}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    {processedTimeSeriesFile ? processedTimeSeriesFile.name : 'Upload File'}
                  </Button>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Pipeline Control</h2>
              <div className="space-y-4">
                <Button
                  className="w-full"
                  onClick={() => runStep(1)}
                  disabled={!rawBuildingsFile && !processedBuildingsFile}
                >
                  <Play className="mr-2 h-4 w-4" />
                  Run Step 1: Process Buildings
                </Button>
                <Button
                  className="w-full"
                  onClick={() => runStep(2)}
                  disabled={!rawTimeSeriesFile && !processedTimeSeriesFile}
                >
                  <Database className="mr-2 h-4 w-4" />
                  Run Step 2: Process Time Series
                </Button>
                <Button
                  className="w-full"
                  onClick={() => runStep(3)}
                  disabled={!buildingsData}
                >
                  <Database className="mr-2 h-4 w-4" />
                  Run Step 3: Generate Assignments
                </Button>
                <Button
                  className="w-full"
                  variant="default"
                  onClick={() => {
                    runStep(1)
                    .then(() => runStep(2))
                    .then(() => runStep(3))
                  }}
                  disabled={!rawBuildingsFile || !rawTimeSeriesFile}
                >
                  Run All Steps
                </Button>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Progress</h2>
              <Progress value={progress} className="mb-2" />
              <p className="text-sm text-gray-600">Processing: {progress}%</p>
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-8">
            <div className="bg-white rounded-lg shadow">
              <Tabs defaultValue="buildings" className="w-full">
                <TabsList className="w-full border-b">
                  <TabsTrigger value="buildings" className="flex-1">Buildings</TabsTrigger>
                  <TabsTrigger value="timeseries" className="flex-1">Time Series</TabsTrigger>
                  <TabsTrigger value="assignments" className="flex-1">Assignments</TabsTrigger>
                </TabsList>
                <TabsContent value="buildings" className="p-6">
                  {buildingsData && buildingsColumns.length > 0 ? (
                    <DataTable columns={buildingsColumns} data={buildingsData} />
                  ) : (
                    <div className="text-center text-gray-500">
                      <p>No building data processed yet.</p>
                      <p className="text-sm">Upload raw buildings CSV and run Step 1 to see results.</p>
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="timeseries" className="p-6">
                  {timeSeriesData && timeSeriesColumns.length > 0 ? (
                    <DataTable columns={timeSeriesColumns} data={timeSeriesData} />
                  ) : (
                    <div className="text-center text-gray-500">
                      <p>No time series data processed yet.</p>
                      <p className="text-sm">Upload raw time series CSV and run Step 2 to see results.</p>
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="assignments" className="p-6">
                  {assignmentsData && assignmentsColumns.length > 0 ? (
                    <DataTable columns={assignmentsColumns} data={assignmentsData} />
                  ) : (
                    <div className="text-center text-gray-500">
                      <p>No assignments generated yet.</p>
                      <p className="text-sm">Run Step 3 to generate building assignments.</p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}