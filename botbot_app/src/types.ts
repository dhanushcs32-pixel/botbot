/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Attachment {
  id: string;
  name: string;
  type: "image" | "pdf" | "text" | "docx";
  size: string;
  url: string;         // Object URL for browser preview rendering
  base64?: string;     // Base64 representation for sending to APIs
  textContents?: string; // Parsed text content for PDFs or text files
}

export interface ChatMessage {
  id: string;
  sender: "user" | "bot";
  text: string;
  timestamp: number;
  status: "sent" | "delivered" | "read";
  attachments?: Attachment[];
  replyTo?: {
    id: string;
    sender: "user" | "bot";
    text: string;
  };
}

export interface ChatSession {
  id: string;
  name: string;
  avatar: string;
  systemPrompt: string;
  messages: ChatMessage[];
  unreadCount: number;
}

export interface WallpaperConfig {
  type: "preset" | "solid" | "custom";
  value: string; // Color code, class name, preset name, or base64 / URL
}
