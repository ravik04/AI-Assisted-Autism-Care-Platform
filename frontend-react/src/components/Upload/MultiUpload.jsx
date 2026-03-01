import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileImage, FileVideo, FileAudio, File } from 'lucide-react';

const ICON_MAP = {
  image: FileImage,
  video: FileVideo,
  audio: FileAudio,
};

function getFileIcon(file) {
  const type = file.type.split('/')[0];
  return ICON_MAP[type] || File;
}

export default function MultiUpload({ onFiles, accept, maxFiles = 10, label = 'Upload files' }) {
  const [files, setFiles] = useState([]);

  const onDrop = useCallback((accepted) => {
    const updated = [...files, ...accepted].slice(0, maxFiles);
    setFiles(updated);
    onFiles(updated);
  }, [files, maxFiles, onFiles]);

  const removeFile = (index) => {
    const updated = files.filter((_, i) => i !== index);
    setFiles(updated);
    onFiles(updated);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: maxFiles - files.length,
    disabled: files.length >= maxFiles,
  });

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
          ${isDragActive ? 'dropzone-active border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-300 hover:bg-slate-50'}
          ${files.length >= maxFiles ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
        <p className="text-sm font-medium text-slate-600">{label}</p>
        <p className="text-xs text-slate-400 mt-1">
          {isDragActive ? 'Drop files here' : `Drag & drop or click to browse (${files.length}/${maxFiles})`}
        </p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, i) => {
            const Icon = getFileIcon(file);
            return (
              <div key={`${file.name}-${i}`} className="flex items-center gap-3 p-2.5 bg-white rounded-lg border border-slate-200">
                <Icon className="w-5 h-5 text-indigo-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-700 truncate">{file.name}</p>
                  <p className="text-xs text-slate-400">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
                <button onClick={() => removeFile(i)} className="p-1 rounded hover:bg-red-50 text-slate-400 hover:text-red-500">
                  <X className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
