"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import Papa from "papaparse";
import clsx from "clsx";

interface UploadStatus {
  success: boolean;
  message: string;
}

interface MatrixData {
  nodeNames: string[];
  matrix: number[][];
}

interface UploadSectionProps {
  onFileUpload: (data: MatrixData) => void;
  uploadStatus: UploadStatus;
  setUploadStatus: (status: UploadStatus) => void;
}

export const UploadSection = ({
  onFileUpload,
  uploadStatus,
  setUploadStatus,
}: UploadSectionProps) => {
  const [filename, setFilename] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setFilename(file.name);

      Papa.parse(file, {
        complete: (result) => {
          try {
            const data = result.data as string[][];

            // Filter out empty rows
            const filteredData = data.filter((row) =>
              row.some((cell) => cell && cell.trim() !== "")
            );

            if (filteredData.length < 2) {
              throw new Error("Matrix must have at least 2 rows");
            }

            // First row contains node names (skip first empty cell)
            const nodeNames = filteredData[0].slice(1).filter(Boolean);

            // Build matrix from remaining rows
            const matrix: number[][] = [];
            for (let i = 1; i < filteredData.length; i++) {
              const row = filteredData[i];
              if (!row[0] || row[0].trim() === "") continue;

              const values = row.slice(1).map((val) => {
                const num = parseInt(val, 10);
                return isNaN(num) ? 0 : num;
              });
              matrix.push(values);
            }

            // Validate square matrix
            if (matrix.length !== nodeNames.length) {
              throw new Error(
                `Matrix not square: ${matrix.length} rows x ${nodeNames.length} columns`
              );
            }

            setUploadStatus({
              success: true,
              message: `${nodeNames.length} nodes loaded`,
            });

            onFileUpload({ nodeNames, matrix });
          } catch (error) {
            setUploadStatus({
              success: false,
              message: error instanceof Error ? error.message : "Parse error",
            });
          }
        },
        error: (error) => {
          setUploadStatus({
            success: false,
            message: error.message,
          });
        },
        skipEmptyLines: true,
      });
    },
    [onFileUpload, setUploadStatus]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  return (
    <div className="terminal-window">
      <div className="terminal-window-header">&gt; LOAD ADJACENCY MATRIX</div>

      <div
        {...getRootProps()}
        className={clsx(
          "border border-dashed border-terminal-green-muted p-8 text-center cursor-pointer",
          "hover:border-terminal-amber hover:bg-terminal-green/5",
          "transition-colors duration-100",
          isDragActive && "border-terminal-amber bg-terminal-green/10"
        )}
      >
        <input {...getInputProps()} />
        <Upload
          className="mx-auto mb-4 w-12 h-12 text-terminal-green"
          strokeWidth={1}
        />
        <p className="text-terminal-green uppercase text-sm mb-2">
          {isDragActive ? "DROP FILE HERE" : "DROP .CSV FILE HERE"}
        </p>
        <p className="text-terminal-green opacity-50 text-xs mb-4">OR</p>
        <button type="button" className="btn-terminal">
          [ BROWSE FILES ]
        </button>
      </div>

      {filename && (
        <div className="mt-4 font-mono text-sm">
          <p className="prompt">upload {filename}</p>
          <p className={uploadStatus.success ? "status-ok" : "status-err"}>
            {uploadStatus.message}
          </p>
        </div>
      )}
    </div>
  );
};
