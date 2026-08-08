import type { ChatMessage, ResearchScope } from "@filing-room/contracts";

export interface LocalConversation {
  id: string;
  title: string;
  updatedAt: string;
  scope: ResearchScope;
  messages: ChatMessage[];
}

const DB_NAME = "filing-room";
const STORE = "conversations";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function listConversations(): Promise<LocalConversation[]> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE).objectStore(STORE).getAll();
    request.onsuccess = () => resolve((request.result as LocalConversation[]).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, 20));
    request.onerror = () => reject(request.error);
  });
}

export async function saveConversation(conversation: LocalConversation): Promise<void> {
  const db = await openDatabase();
  const existing = await listConversations();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).put(conversation);
    for (const old of existing.slice(19)) transaction.objectStore(STORE).delete(old.id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

export async function deleteConversation(id: string): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).delete(id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

export async function clearConversations(): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).clear();
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}
