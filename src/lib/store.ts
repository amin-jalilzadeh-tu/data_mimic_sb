import { create } from 'zustand'

interface FileState {
  rawBuildingsFile: File | null
  rawTimeSeriesFile: File | null
  processedBuildingsFile: File | null
  processedTimeSeriesFile: File | null
  buildingsData: any[] | null
  timeSeriesData: any[] | null
  assignmentsData: any[] | null
  progress: number
  status: string
  setRawBuildingsFile: (file: File | null) => void
  setRawTimeSeriesFile: (file: File | null) => void
  setProcessedBuildingsFile: (file: File | null) => void
  setProcessedTimeSeriesFile: (file: File | null) => void
  setBuildingsData: (data: any[] | null) => void
  setTimeSeriesData: (data: any[] | null) => void
  setAssignmentsData: (data: any[] | null) => void
  setProgress: (progress: number) => void
  setStatus: (status: string) => void
}

export const useStore = create<FileState>((set) => ({
  rawBuildingsFile: null,
  rawTimeSeriesFile: null,
  processedBuildingsFile: null,
  processedTimeSeriesFile: null,
  buildingsData: null,
  timeSeriesData: null,
  assignmentsData: null,
  progress: 0,
  status: '',
  setRawBuildingsFile: (file) => set({ rawBuildingsFile: file }),
  setRawTimeSeriesFile: (file) => set({ rawTimeSeriesFile: file }),
  setProcessedBuildingsFile: (file) => set({ processedBuildingsFile: file }),
  setProcessedTimeSeriesFile: (file) => set({ processedTimeSeriesFile: file }),
  setBuildingsData: (data) => set({ buildingsData: data }),
  setTimeSeriesData: (data) => set({ timeSeriesData: data }),
  setAssignmentsData: (data) => set({ assignmentsData: data }),
  setProgress: (progress) => set({ progress }),
  setStatus: (status) => set({ status }),
}))