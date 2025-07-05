import os
import json
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
from config import Config
import logging
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import re

logger = logging.getLogger(__name__)


def fix_metadata_none_values(metadata_dict):
    """Fix None values in metadata to be ChromaDB-compatible."""
    fixed_metadata = {}
    for key, value in metadata_dict.items():
        if value is None:
            fixed_metadata[key] = ""
        elif isinstance(value, (int, float, bool)):
            fixed_metadata[key] = value
        else:
            fixed_metadata[key] = str(value)
    return fixed_metadata



def extract_section_headers(text: str) -> list:
    """
    Extract section headers robustly for all 6 companies: find the last TOC-style line and extract headers after that line.
    Handles boxed narrative headers (│  Item X. ...), standard, all-caps, and boxed (multi-line) headers.
    Always sets a valid PART for each section, inferring from item number if not set by a PART header.
    Returns list of (start_line, part, item, item_title) tuples, where start_line is the line number (not char offset).
    """
    headers = []
    lines = text.splitlines()
    current_part = None
    # Find the last TOC-style line (e.g., 'Item X. ... <page number>')
    TOC_LINE_REGEX = re.compile(r'^\s*Item\s+[0-9]+[A-Z]?\.?.*\s+\d+\s*$')
    last_toc_line = -1
    for i, line in enumerate(lines):
        if TOC_LINE_REGEX.match(line.strip()):
            last_toc_line = i
    # Regex for boxed narrative headers (│  Item X. ... or |  Item X. ...)
    BOXED_NARRATIVE_HEADER_REGEX = re.compile(r'^[│|] *Item +([0-9]+[A-Z]?)\.?(.*?)$', re.IGNORECASE)
    PART_HEADER_REGEX = re.compile(r'^\s*P\s*A\s*R\s*T\s+(I|II|III|IV)\s*$', re.IGNORECASE)
    ITEM_STANDARD_REGEX = re.compile(r'^\s*Item\s+([0-9]+[A-Z]?)\.\s+(.+?)\s*$', re.IGNORECASE)
    ITEM_ALL_CAPS_REGEX = re.compile(r'^\s*ITEM\s+([0-9]+[A-Z]?)\.\s+(.+?)\s*$')
    ITEM_BOXED_REGEX = re.compile(r'^\s*Item\s+([0-9]+[A-Z]?)\.?\s*$', re.IGNORECASE)
    TITLE_BOXED_REGEX = re.compile(r'^\s*([A-Za-z][A-Za-z\s]+?)\s*$')
    def infer_part(item_num):
        # Infer PART from item number (10-K convention)
        try:
            # Remove trailing letters (e.g., 1A -> 1)
            base = re.match(r'(\d+)', item_num)
            if not base:
                return ""
            n = int(base.group(1))
            if 1 <= n <= 4:
                return "Part I"
            elif 5 <= n <= 9:
                return "Part II"
            elif 10 <= n <= 14:
                return "Part III"
            elif 15 <= n <= 16:
                return "Part IV"
        except Exception:
            return ""
        return ""
    for i in range(last_toc_line + 1, len(lines)):
        line_stripped = lines[i].strip()
        # 1. Boxed narrative header (│  Item X. ... or |  Item X. ...)
        boxed_match = BOXED_NARRATIVE_HEADER_REGEX.match(line_stripped)
        if boxed_match:
            item_num = boxed_match.group(1)
            item_title = boxed_match.group(2).strip()
            item_title = re.sub(r'[│|╭╰╔╚•─—═§]+$', '', item_title).strip()
            part = current_part or infer_part(item_num)
            headers.append((i, part, f"Item {item_num}", item_title))
            continue
        # 2. PART header
        part_match = PART_HEADER_REGEX.match(line_stripped)
        if part_match:
            current_part = f"Part {part_match.group(1)}"
            continue
        # 3. Standard Item format
        item_match = ITEM_STANDARD_REGEX.match(line_stripped)
        if item_match:
            item_num = item_match.group(1)
            item_title = item_match.group(2).strip()
            part = current_part or infer_part(item_num)
            headers.append((i, part, f"Item {item_num}", item_title))
            continue
        # 4. All caps format
        item_caps_match = ITEM_ALL_CAPS_REGEX.match(line_stripped)
        if item_caps_match:
            item_num = item_caps_match.group(1)
            item_title = item_caps_match.group(2).strip()
            part = current_part or infer_part(item_num)
            headers.append((i, part, f"Item {item_num}", item_title))
            continue
        # 5. Boxed (multi-line) format
        item_boxed_match = ITEM_BOXED_REGEX.match(line_stripped)
        if item_boxed_match:
            item_num = item_boxed_match.group(1)
            j = i + 1
            item_title = None
            while j < len(lines) and j - i <= 5:
                next_line = lines[j].strip()
                if next_line and not next_line.startswith(('│', '╭', '╰', '╔', '╚', '═', '§')):
                    title_match = TITLE_BOXED_REGEX.match(next_line)
                    if title_match:
                        potential_title = title_match.group(1).strip()
                        if (len(potential_title) > 2 and len(potential_title) < 100 and not potential_title.lower() in ['item', 'part', 'section', 'page']):
                            item_title = potential_title
                            break
                j += 1
            part = current_part or infer_part(item_num)
            headers.append((i, part, f"Item {item_num}", item_title))
            continue
    return headers

def extract_section_content(text: str, section_start_line: int, section_end_line: int) -> str:
    """
    Extract the content of a section between start and end line numbers.
    Skips blank and decorative lines after the header, starting at the first real content line.
    """
    lines = text.splitlines()
    if section_end_line == -1 or section_end_line > len(lines):
        section_end_line = len(lines)
    # Scan forward from section_start_line+1 to first real content line
    content_start = section_start_line + 1
    while content_start < section_end_line:
        line = lines[content_start].strip()
        # Skip blank or decorative lines
        if line and not all(c in '│╭╰╔╚•─—═§|= ' for c in line):
            break
        content_start += 1
    return '\n'.join(lines[content_start:section_end_line])

def create_semantic_chunks(text: str, headers: list, max_chunk_size: int = 3000, overlap_size: int = 300) -> list:
    chunks = []
    for i, (start_line, part, item, item_title) in enumerate(headers):
        section_end_line = headers[i+1][0] if i+1 < len(headers) else -1
        section_content = extract_section_content(text, start_line, section_end_line)
        if len(section_content) <= max_chunk_size:
            chunk_data = {
                'content': section_content,
                'start_idx': start_line,
                'end_idx': section_end_line if section_end_line != -1 else len(text.splitlines()),
                'part': part,
                'item': item,
                'item_title': item_title,
                'sub_section': None,
                'chunk_index': 0,
                'total_chunks': 1
            }
            chunks.append(chunk_data)
        else:
            sub_chunks = split_section_into_subchunks(
                section_content, start_line, part, item, item_title, max_chunk_size, overlap_size
            )
            chunks.extend(sub_chunks)
    return chunks

def split_section_into_subchunks(section_content: str, section_start: int, 
                                part: str, item: str, item_title: str,
                                max_chunk_size: int, overlap_size: int) -> List[Dict]:
    """
    Split a long section into sub-chunks based on sub-headings.
    """
    chunks = []
    
    # For the old semantic chunking approach, we'll use simple character-based splitting
    # since we don't need the complex sub-heading detection anymore
    return split_by_character_count(
        section_content, section_start, part, item, item_title,
        max_chunk_size, overlap_size
    )

def split_by_character_count(section_content: str, section_start: int,
                           part: str, item: str, item_title: str,
                           max_chunk_size: int, overlap_size: int) -> List[Dict]:
    """
    Fallback: split section by character count with overlap.
    """
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(section_content):
        end = start + max_chunk_size
        
        # Try to break at a sentence boundary
        if end < len(section_content):
            # Look for sentence endings near the break point
            for i in range(end, max(end - 200, start), -1):
                if section_content[i] in '.!?':
                    end = i + 1
                    break
        
        chunk_content = section_content[start:end].strip()
        if chunk_content:
            chunks.append({
                'content': chunk_content,
                'start_idx': section_start + start,
                'end_idx': section_start + end,
                'part': part,
                'item': item,
                'item_title': item_title,
                'sub_section': None,
                'chunk_index': chunk_index,
                'total_chunks': (len(section_content) + max_chunk_size - 1) // max_chunk_size
            })
            chunk_index += 1
        
        start = end - overlap_size
    
    return chunks

class SECVectorStore:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding function
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.config.EMBEDDING_MODEL)
        
        # Get or create collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
    def create_metadata(self, filing_info: Dict, chunk_index: int, total_chunks: int, 
                       chunk_type: str = "text") -> Dict[str, Any]:
        filing_date = filing_info.get("filing_date")
        year = ""
        if filing_date and len(filing_date) >= 4:
            year = filing_date[:4]
        company_name = filing_info.get("company_name", "")
        fiscal_year = filing_info.get("fiscal_year", "")
        # Create source file name from ticker and accession number
        ticker = filing_info.get("ticker", "")
        accession_number = filing_info.get("accession_number", "")
        source_file = f"{ticker}_{accession_number}.txt" if ticker and accession_number else ""
        # Ensure all values are ChromaDB-compatible (no None values)
        return {
            "ticker": ticker,
            "company_name": company_name if company_name else "",
            "accession_number": accession_number,
            "filing_date": filing_date if filing_date else "",
            "year": year if year else "",
            "fiscal_year": fiscal_year if fiscal_year else "",
            "form": filing_info.get("form", ""),
            "filing_type": filing_info.get("form", ""),  # Alias for consistency
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_type": chunk_type,
            "file_number": filing_info.get("file_number", ""),
            "source_file": source_file
        }
    
    def parse_structured_sections(self, text_content: str) -> List[Dict]:
        """Parse text content with structured sections (--- SECTION --- format)"""
        sections = []
        lines = text_content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if this is a section header
            if line.strip().startswith('--- ') and line.strip().endswith(' ---'):
                # Save previous section if exists
                if current_section and current_content:
                    sections.append({
                        'section_name': current_section,
                        'content': '\n'.join(current_content).strip(),
                        'section_type': self.get_section_type(current_section)
                    })
                
                # Start new section
                current_section = line.strip()[4:-4].strip()  # Remove "--- " and " ---"
                current_content = []
            else:
                # Add line to current section content
                if current_section is not None:
                    current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append({
                'section_name': current_section,
                'content': '\n'.join(current_content).strip(),
                'section_type': self.get_section_type(current_section)
            })
        
        return sections
    
    def get_section_type(self, section_name: str) -> str:
        """Map section names to standardized types"""
        section_name_lower = section_name.lower()
        
        # Exact matches for common section names
        if section_name_lower == 'business':
            return 'business_description'
        elif section_name_lower == 'risk factors':
            return 'risk_factors'
        elif section_name_lower == "management's discussion and analysis":
            return 'management_discussion'
        elif section_name_lower == 'balance sheet':
            return 'balance_sheet'
        elif section_name_lower == 'income statement':
            return 'income_statement'
        elif section_name_lower == 'operations statement':
            return 'operations_statement'
        elif section_name_lower == 'cash flow statement':
            return 'cash_flow_statement'
        elif section_name_lower == 'comprehensive income':
            return 'comprehensive_income'
        elif section_name_lower == 'stockholders equity':
            return 'stockholders_equity'
        
        # Partial matches for variations
        elif 'business' in section_name_lower:
            return 'business_description'
        elif 'risk' in section_name_lower:
            return 'risk_factors'
        elif 'management' in section_name_lower and 'discussion' in section_name_lower:
            return 'management_discussion'
        elif 'balance' in section_name_lower and 'sheet' in section_name_lower:
            return 'balance_sheet'
        elif 'income' in section_name_lower and 'statement' in section_name_lower:
            return 'income_statement'
        elif 'operations' in section_name_lower:
            return 'operations_statement'
        elif 'cash' in section_name_lower and 'flow' in section_name_lower:
            return 'cash_flow_statement'
        elif 'comprehensive' in section_name_lower and 'income' in section_name_lower:
            return 'comprehensive_income'
        elif ('stockholders' in section_name_lower or 'shareholders' in section_name_lower) and 'equity' in section_name_lower:
            return 'stockholders_equity'
        else:
            return 'other'
    
    def chunk_structured_section(self, section: Dict, max_chunk_size: int = 3000, overlap_size: int = 300) -> List[Dict]:
        """Chunk a structured section into smaller pieces with better sentence boundary preservation"""
        chunks = []
        content = section['content']
        
        if len(content) <= max_chunk_size:
            # Section is small enough to keep as one chunk
            chunks.append({
                'content': content,
                'section_name': section['section_name'],
                'section_type': section['section_type'],
                'chunk_index': 0,
                'total_chunks': 1
            })
        else:
            # Split into multiple chunks with sentence boundary preservation
            sentences = self.split_into_sentences(content)
            current_chunk = []
            current_size = 0
            chunk_index = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                if current_size + sentence_size > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_content = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_content,
                        'section_name': section['section_name'],
                        'section_type': section['section_type'],
                        'chunk_index': chunk_index,
                        'total_chunks': -1  # Will be updated later
                    })
                    
                    # Start new chunk with overlap (last few sentences)
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s) for s in current_chunk)
                    chunk_index += 1
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Add the last chunk
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'section_name': section['section_name'],
                    'section_type': section['section_type'],
                    'chunk_index': chunk_index,
                    'total_chunks': -1  # Will be updated later
                })
            
            # Update total_chunks for all chunks
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk['total_chunks'] = total_chunks
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving formatting"""
        import re
        
        # Split on sentence endings but preserve some formatting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def add_filing_text(self, filing_info: Dict, text_content: str) -> int:
        """Add text filing content to vector store using structured section parsing."""
        try:
            # Check if this is a structured text file (with --- SECTION --- headers)
            if '--- BUSINESS ---' in text_content or '--- RISK FACTORS ---' in text_content:
                # Use structured section parsing
                sections = self.parse_structured_sections(text_content)
                
                if not sections:
                    logger.warning(f"No structured sections found for {filing_info['ticker']} {filing_info['accession_number']}")
                    return 0
                
                # Create chunks from structured sections
                max_chunk_size = getattr(self.config, 'MAX_CHUNK_SIZE', 3000)
                overlap_size = getattr(self.config, 'CHUNK_OVERLAP', 300)
                
                all_chunks = []
                for section in sections:
                    section_chunks = self.chunk_structured_section(section, max_chunk_size, overlap_size)
                    all_chunks.extend(section_chunks)
                
            else:
                # Fall back to the original semantic chunking for raw text files
                logger.info(f"Using original semantic chunking for {filing_info['ticker']} {filing_info['accession_number']}")
                headers = extract_section_headers(text_content)
                
                if not headers:
                    logger.warning(f"No section headers found for {filing_info['ticker']} {filing_info['accession_number']}")
                    return 0

                # Create semantic chunks based on sections and sub-sections
                max_chunk_size = getattr(self.config, 'MAX_CHUNK_SIZE', 3000)
                overlap_size = getattr(self.config, 'CHUNK_OVERLAP', 300)
                
                all_chunks = create_semantic_chunks(text_content, headers, max_chunk_size, overlap_size)

            # Prepare documents for insertion
            documents = []
            metadatas = []
            ids = []

            for i, chunk_data in enumerate(all_chunks):
                chunk_id = f"{filing_info['accession_number']}_text_{i}"
                
                meta = self.create_metadata(filing_info, i, len(all_chunks), "text")
                
                # Add section-specific metadata
                if 'section_name' in chunk_data:
                    # Structured section format
                    meta["section_name"] = chunk_data.get("section_name", "") or ""
                    meta["section_type"] = chunk_data.get("section_type", "") or ""
                    meta["part"] = self.map_section_to_part(chunk_data.get("section_type", ""))
                    meta["item"] = self.map_section_to_item(chunk_data.get("section_type", ""))
                    meta["item_title"] = chunk_data.get("section_name", "") or ""
                else:
                    # Original semantic chunking format
                    meta["part"] = chunk_data.get("part", "") or ""
                    meta["item"] = chunk_data.get("item", "") or ""
                    meta["item_title"] = chunk_data.get("item_title", "") or ""
                    meta["section_name"] = chunk_data.get("item_title", "") or ""
                    meta["section_type"] = self.get_section_type_from_item(chunk_data.get("item", ""))
                
                meta["sub_section"] = chunk_data.get("sub_section", "") or ""
                meta["chunk_index"] = chunk_data.get("chunk_index", 0)
                meta["total_chunks"] = chunk_data.get("total_chunks", 1)
                
                # Add enhanced metadata for better filtering
                meta["section_title"] = meta.get("item_title", "") or meta.get("section_name", "")
                meta["filing_type"] = filing_info.get("form", "")
                
                # Create source file name
                ticker = filing_info.get("ticker", "")
                accession_number = filing_info.get("accession_number", "")
                meta["source_file"] = f"{ticker}_{accession_number}.txt" if ticker and accession_number else ""
                
                # Fix any None values in metadata
                meta = fix_metadata_none_values(meta)
                documents.append(chunk_data["content"])
                metadatas.append(meta)
                ids.append(chunk_id)

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            # Calculate statistics
            total_chunks_added = len(documents)
            
            # Count unique sections
            unique_sections = set()
            for meta in metadatas:
                section_name = meta.get("section_name", "")
                if section_name:
                    unique_sections.add(section_name)
            
            # Group by section type
            section_types_summary = {}
            for meta in metadatas:
                section_type = meta.get("section_type", "")
                if section_type:
                    if section_type not in section_types_summary:
                        section_types_summary[section_type] = 0
                    section_types_summary[section_type] += 1
            
            # Write detailed report to file
            report_lines = []
            report_lines.append(f"=== STRUCTURED INGESTION SUMMARY FOR {filing_info['ticker']} {filing_info['accession_number']} ===")
            report_lines.append(f"Total chunks added: {total_chunks_added}")
            report_lines.append(f"Unique sections: {len(unique_sections)}")
            report_lines.append(f"Max chunk size: {max_chunk_size}")
            report_lines.append(f"Overlap size: {overlap_size}")
            
            report_lines.append(f"\nSection breakdown:")
            for section_type, count in sorted(section_types_summary.items()):
                report_lines.append(f"  {section_type}: {count} chunks")
            
            report_lines.append(f"\nUnique sections found:")
            for section in sorted(unique_sections):
                report_lines.append(f"  {section}")
            
            # Write to file
            with open("Ingestion_report.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(report_lines) + "\n\n" + "="*80 + "\n\n")

            logger.info(f"Added {len(documents)} chunks for {filing_info['ticker']} {filing_info['accession_number']}")
            return len(documents)

        except Exception as e:
            logger.error(f"Error adding filing text to vector store: {str(e)}")
            return 0
    
    def map_section_to_part(self, section_type: str) -> str:
        """Map section type to 10-K part"""
        mapping = {
            'business_description': 'Part I',
            'risk_factors': 'Part I',
            'management_discussion': 'Part II',
            'balance_sheet': 'Part II',
            'income_statement': 'Part II',
            'operations_statement': 'Part II',
            'cash_flow_statement': 'Part II',
            'comprehensive_income': 'Part II',
            'stockholders_equity': 'Part II',
            'financial_statements': 'Part II',
            'other': 'Part I'
        }
        return mapping.get(section_type, 'Part I')
    
    def map_section_to_item(self, section_type: str) -> str:
        """Map section type to 10-K item"""
        mapping = {
            'business_description': 'Item 1',
            'risk_factors': 'Item 1A',
            'management_discussion': 'Item 7',
            'balance_sheet': 'Item 8',
            'income_statement': 'Item 8',
            'operations_statement': 'Item 8',
            'cash_flow_statement': 'Item 8',
            'comprehensive_income': 'Item 8',
            'stockholders_equity': 'Item 8',
            'financial_statements': 'Item 8',
            'other': 'Item 1'
        }
        return mapping.get(section_type, 'Item 1')
    
    def get_section_type_from_item(self, item: str) -> str:
        """Get section type from item number"""
        mapping = {
            'Item 1': 'business_description',
            'Item 1A': 'risk_factors',
            'Item 7': 'management_discussion',
            'Item 8': 'financial_statements'
        }
        return mapping.get(item, 'other')
    
    def add_financial_facts(self, filing_info: Dict, facts: Dict[str, Any]) -> int:
        """Add structured financial facts to vector store"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            # Handle the new XBRL facts structure
            if 'financial_facts' in facts:
                # New structure from XBRL extraction
                financial_facts = facts['financial_facts']
                for fact_key, fact_data in financial_facts.items():
                    # Create a structured text representation of the fact
                    fact_text = f"Financial Fact: {fact_data.get('element', fact_key)}\n"
                    fact_text += f"Value: {fact_data.get('value', 'N/A')}\n"
                    fact_text += f"Context: {fact_data.get('context', 'N/A')}\n"
                    fact_text += f"Unit: {fact_data.get('unit', 'N/A')}\n"
                    fact_text += f"Filing: {filing_info.get('form', 'N/A')} - {filing_info.get('filing_date', 'N/A')}"
                    
                    fact_id = f"{filing_info['accession_number']}_fact_{fact_key}"
                    
                    documents.append(fact_text)
                    
                    # Create metadata with XBRL-specific fields
                    metadata = self.create_metadata(filing_info, 0, 1, "financial_fact")
                    metadata.update({
                        'element': fact_data.get('element', ''),
                        'label': fact_data.get('label', ''),
                        'context': fact_data.get('context', ''),
                        'statement_type': fact_data.get('statement_type', ''),
                        'unit': fact_data.get('unit', '')
                    })
                    metadata = fix_metadata_none_values(metadata)
                    metadatas.append(metadata)
                    ids.append(fact_id)
            else:
                # Old structure (backward compatibility)
                for fact_key, fact_data in facts.items():
                    # Create a structured text representation of the fact
                    fact_text = f"Financial Fact: {fact_key}\n"
                    fact_text += f"Value: {fact_data.get('value', 'N/A')}\n"
                    fact_text += f"Date: {fact_data.get('date', 'N/A')}\n"
                    fact_text += f"Form: {fact_data.get('form', 'N/A')}"
                    
                    fact_id = f"{filing_info['accession_number']}_fact_{fact_key}"
                    
                    documents.append(fact_text)
                    metadata = self.create_metadata(filing_info, 0, 1, "financial_fact")
                    metadata = fix_metadata_none_values(metadata)
                    metadatas.append(metadata)
                    ids.append(fact_id)
            
            # Add to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
            logger.info(f"Added {len(documents)} financial facts for {filing_info['ticker']} {filing_info['accession_number']}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding financial facts to vector store: {str(e)}")
            return 0
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def search_by_company(self, query: str, ticker: str, n_results: int = 5) -> List[Dict]:
        """Search for documents from a specific company"""
        return self.search(query, n_results, {"ticker": ticker})
    
    def search_by_filing_type(self, query: str, form: str, n_results: int = 5) -> List[Dict]:
        """Search for documents from a specific filing type"""
        return self.search(query, n_results, {"form": form})
    
    def get_filing_summary(self, accession_number: str) -> List[Dict]:
        """Get all chunks for a specific filing"""
        return self.search("", n_results=1000, filter_dict={"accession_number": accession_number})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            
            # Get unique companies and filings
            all_metadata = self.collection.get()
            companies = set()
            filings = set()
            
            for metadata in all_metadata['metadatas']:
                companies.add(metadata.get('ticker', ''))
                filings.add(metadata.get('accession_number', ''))
            
            return {
                'total_documents': count,
                'unique_companies': len(companies),
                'unique_filings': len(filings),
                'companies': list(companies)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=SentenceTransformerEmbeddingFunction(model_name=self.config.EMBEDDING_MODEL),
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}") 