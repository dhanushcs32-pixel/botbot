/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Extracts printable ASCII / UTF-8 text from raw PDF ArrayBuffer streams.
 * This is a lightweight, pure-JS client-side PDF parser that reads stream structures.
 */
export function extractTextFromPDF(arrayBuffer: ArrayBuffer): string {
  try {
    const uint8 = new Uint8Array(arrayBuffer);
    let binary = "";
    
    // Read up to 1MB of the file to keep performance high
    const length = Math.min(uint8.length, 1000000);
    for (let i = 0; i < length; i++) {
      binary += String.fromCharCode(uint8[i]);
    }
    
    // Find text objects inside stream segments
    const streams = binary.match(/stream[\s\S]*?endstream/g) || [];
    let extractedText = "";
    
    for (const stream of streams) {
      // Find segments wrapped in parentheses which represent PDF strings: (String Text)
      const matches = stream.match(/\(([^)]+)\)/g);
      if (matches) {
        for (const match of matches) {
          const clean = match.slice(1, -1)
            .replace(/\\([\s\S])/g, "$1") // unescape PDF control slashes
            .trim();
          
          // Filter out PDF internal names, fonts, or garbage non-text streams
          if (
            clean.length > 1 && 
            !clean.startsWith("/") && 
            !/^[a-zA-Z0-9]{30,}$/.test(clean) &&
            !clean.includes("font") &&
            !clean.includes("Adobe")
          ) {
            extractedText += clean + " ";
          }
        }
      }
    }
    
    // Fallback search if stream-matching failed (e.g., uncompressed flat PDFs)
    if (extractedText.trim().length < 15) {
      const allMatches = binary.match(/\(([^)]+)\)/g) || [];
      let count = 0;
      for (const match of allMatches) {
        const clean = match.slice(1, -1).trim();
        if (
          clean.length > 2 && 
          !clean.startsWith("/") && 
          !/^[a-zA-Z0-9]{30,}$/.test(clean) &&
          !clean.includes("font") && 
          !clean.includes("Widths") &&
          count < 1500
        ) {
          extractedText += clean + " ";
          count++;
        }
      }
    }
    
    // Clean up double spaces, control characters, and return
    return extractedText
      .replace(/[\r\n\t]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  } catch (err) {
    console.error("Error in PDF client parser:", err);
    return "";
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
