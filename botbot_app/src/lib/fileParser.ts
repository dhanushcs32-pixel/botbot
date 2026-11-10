/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Safely inspects or marks PDF ArrayBuffers without blocking the UI thread or regex-scanning binary streams.
 * Full PDF text extraction is processed server-side via pypdf/pdfplumber in media.py.
 */
export function extractTextFromPDF(arrayBuffer: ArrayBuffer, fileName: string = "document.pdf"): string {
  try {
    const size = arrayBuffer ? arrayBuffer.byteLength : 0;
    return `[PDF attached: ${fileName} (${formatBytes(size)}) - text extraction is handled server-side]`;
  } catch (err) {
    console.error("Error inspecting PDF attachment:", err);
    return `[PDF attached: ${fileName}]`;
  }
}

/**
 * Format bytes into a human-readable size
 */
export function formatBytes(bytes: number, decimals = 1): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}
