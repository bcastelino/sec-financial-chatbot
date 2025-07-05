import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
try:
    import streamlit as st
except ImportError:
    # Create a mock st object for testing outside Streamlit
    class MockStreamlit:
        def error(self, message):
            print(f"ERROR: {message}")
    st = MockStreamlit()
from config import Config

class SECChatbot:
    def __init__(self):
        """
        SECChatbot: Implements a RAG (Retrieval-Augmented Generation) pipeline for SEC filings.
        Steps:
        1. Documents are embedded and indexed in a vector database (ChromaDB).
        2. User queries are embedded using the same model.
        3. Similarity search retrieves relevant document chunks.
        4. Retrieved context is sent to the LLM as part of the prompt.
        5. LLM generates a final response.
        """
        # --- RAG Step 1: Vector DB and Embedding Model Setup ---
        self.openrouter_api_key = Config.OPENROUTER_API_KEY
        self.model = Config.DEFAULT_MODEL
        self.base_url = Config.OPENROUTER_BASE_URL
        # ChromaDB stores document embeddings (already indexed by ingest script)
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection(Config.COLLECTION_NAME)
        self.chat_history = []

    def query_vector_db(self, query: str, n_results: int = None) -> Dict[str, List]:
        """
        RAG Step 2-3: Query embedding and similarity search.
        - Encode the user query (embedding model is managed by ChromaDB backend).
        - Retrieve the most similar document chunks from the vector DB.
        """
        import re
        if n_results is None:
            n_results = Config.DEFAULT_NUM_TEXT_CHUNKS if hasattr(Config, 'DEFAULT_NUM_TEXT_CHUNKS') else 50
        filter_dict = None
        tickers = [c["ticker"] for c in Config.TOP_COMPANIES]
        ticker_in_query = next((t for t in tickers if t.lower() in query.lower()), None)
        year_match = re.search(r"20[0-9]{2}", query)
        year = year_match.group(0) if year_match else None
        if ticker_in_query or year:
            filter_dict = {}
            if ticker_in_query:
                filter_dict["ticker"] = ticker_in_query
            if year:
                all_metadata = self.collection.get(include=["metadatas"])
                filing_dates = set()
                for meta in all_metadata["metadatas"]:
                    if meta and "filing_date" in meta and str(meta["filing_date"]).startswith(year):
                        filing_dates.add(meta["filing_date"])
                if filing_dates:
                    filter_dict["filing_date"] = {"$in": list(filing_dates)}
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            return {"text_results": results}
        except Exception as e:
            st.error(f"Error querying vector database: {e}")
            return {"text_results": []}

    def format_context(self, query_results: Dict[str, List]) -> str:
        """
        RAG Step 4: Format retrieved context for the LLM prompt.
        - Combine relevant document chunks and metadata for the LLM.
        """
        context_parts = []
        if query_results.get("text_results") and query_results["text_results"].get("documents"):
            context_parts.append("=== RELEVANT FILING SECTIONS ===")
            for i, doc in enumerate(query_results["text_results"]["documents"][0]):
                metadata = query_results["text_results"]["metadatas"][0][i]
                company = metadata.get("company_name", "Unknown")
                filing_type = metadata.get("form", "Unknown")
                date = metadata.get("filing_date", "Unknown")
                chunk_type = metadata.get("chunk_type", "text")
                context_parts.append(f"\n--- {company} {filing_type} ({date}) [{chunk_type}] ---")
                context_parts.append(doc[:1000] + "..." if len(doc) > 1000 else doc)
        return "\n".join(context_parts)

    def call_openrouter_api(self, messages: List[Dict[str, str]]) -> str:
        """
        RAG Step 5: Call the LLM with the context-augmented prompt.
        - The LLM (via OpenRouter API) generates the final answer.
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sec-chatbot.streamlit.app",
            "X-Title": "SEC Filing Chatbot"
        }
        payload = {
            "model": Config.DEFAULT_MODEL,
            "messages": messages
        }
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                error_msg = f"OpenRouter API error: {result['error'].get('message', result['error'])}"
                st.error(error_msg)
                return error_msg
            else:
                error_msg = f"Unexpected API response: {result}"
                st.error(error_msg)
                return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"API call failed: {e}"
            st.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            st.error(error_msg)
            return error_msg

    def get_xbrl_facts(self, company_ticker: str, fiscal_year: str) -> List[Dict]:
        """Retrieve all XBRL (financial_fact) chunks for a company and fiscal year."""
        filter_dict = {"$and": [
            {"ticker": company_ticker},
            {"fiscal_year": fiscal_year},
            {"chunk_type": "financial_fact"}
        ]}
        try:
            results = self.collection.get(where=filter_dict, include=["documents", "metadatas"])
            xbrl_chunks = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]
                    xbrl_chunks.append({"document": doc, "metadata": metadata})
            return xbrl_chunks
        except Exception as e:
            st.error(f"Error retrieving XBRL facts: {e}")
            return []

    def get_all_text_chunks(self, company_ticker: str, fiscal_year: str) -> list:
        """Retrieve all text chunks for a company and fiscal year."""
        filter_dict = {"$and": [
            {"ticker": company_ticker},
            {"fiscal_year": fiscal_year},
            {"chunk_type": "text"}
        ]}
        try:
            results = self.collection.get(where=filter_dict, include=["documents", "metadatas"])
            text_chunks = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i]
                    text_chunks.append({"document": doc, "metadata": metadata})
            return text_chunks
        except Exception as e:
            st.error(f"Error retrieving text chunks: {e}")
            return []

    def filter_chunks_by_section(self, text_chunks: list, section_filter: tuple) -> list:
        """Filter text chunks by section type or name"""
        if not section_filter or not text_chunks:
            return text_chunks
        
        item_code, item_title = section_filter
        filtered_chunks = []
        
        # Normalize search terms
        def normalize(s):
            return s.lower().replace(' ', '').replace('.', '').replace('-', '') if s else ''
        
        # Primary search terms (exact matches)
        primary_terms = [
            item_code.lower(),  # "item 1a"
            item_title.lower(),  # "risk factors"
            normalize(item_code),  # "item1a"
            normalize(item_title)  # "riskfactors"
        ]
        
        # Secondary search terms (for broader matching)
        secondary_terms = []
        section_type_mapping = {
            "Item 1A": ["risk_factors", "risk factors", "riskfactors"],
            "Item 1": ["business_description", "business", "businessdescription"],
            "Item 7": ["management_discussion", "management discussion", "md&a", "md&a"],
            "Item 8": ["balance_sheet", "income_statement", "cash_flow_statement", "financial statements", "financial_statements"],
            "Item 7A": ["quantitative", "qualitative", "market risk"]
        }
        
        if item_code in section_type_mapping:
            secondary_terms.extend(section_type_mapping[item_code])
        
        for chunk in text_chunks:
            metadata = chunk["metadata"]
            document = chunk["document"].lower()
            
            # Check metadata fields first (higher priority)
            metadata_fields = [
                metadata.get("section_name", ""),
                metadata.get("section_type", ""),
                metadata.get("item_title", ""),
                metadata.get("item", ""),
                metadata.get("section_title", "")
            ]
            
            # First, try to match primary terms in metadata (exact matches)
            primary_match = False
            for term in primary_terms:
                for field in metadata_fields:
                    if field and term in field.lower():
                        primary_match = True
                        break
                if primary_match:
                    break
            
            # If no primary match, try secondary terms in metadata
            secondary_match = False
            if not primary_match:
                for term in secondary_terms:
                    for field in metadata_fields:
                        if field and term in field.lower():
                            secondary_match = True
                            break
                    if secondary_match:
                        break
            
            # Only check document content if no metadata match found
            content_match = False
            if not primary_match and not secondary_match:
                # Check if the document starts with the item code or title
                doc_start = document[:200].lower()  # Check first 200 chars
                for term in primary_terms:
                    if term in doc_start:
                        content_match = True
                        break
            
            # Add chunk if any match found
            if primary_match or secondary_match or content_match:
                filtered_chunks.append(chunk)
        
        # If no matches found, return a subset of chunks instead of all
        if not filtered_chunks:
            # Return chunks that might be related (first 10)
            return text_chunks[:10]
        
        return filtered_chunks

    def format_context_with_sections(self, text_chunks: list) -> str:
        """Format context with clear section information"""
        if not text_chunks:
            return ""
        
        context_parts = []
        context_parts.append("=== FILING TEXT SECTIONS ===")
        
        # Group chunks by section for better organization
        sections = {}
        for chunk in text_chunks:
            meta = chunk["metadata"]
            section_name = meta.get("section_name", meta.get("item_title", "Unknown Section"))
            section_type = meta.get("section_type", "unknown")
            
            if section_name not in sections:
                sections[section_name] = {
                    'type': section_type,
                    'chunks': [],
                    'metadata': meta
                }
            sections[section_name]['chunks'].append(chunk)
        
        # Format each section
        for section_name, section_data in sections.items():
            meta = section_data['metadata']
            company = meta.get("company_name", "Unknown")
            filing_type = meta.get("form", "Unknown")
            date = meta.get("filing_date", "Unknown")
            section_type = section_data['type']
            
            context_parts.append(f"\n--- {company} {filing_type} ({date}) - {section_name} [{section_type}] ---")
            
            # Combine all chunks for this section
            section_content = ""
            for chunk in section_data['chunks']:
                section_content += chunk["document"] + "\n\n"
            
            # Truncate if too long
            if len(section_content) > 2000:
                section_content = section_content[:2000] + "..."
            
            context_parts.append(section_content)
        
        return "\n".join(context_parts)



    def postprocess_llm_response(self, response: str) -> str:
        """Remove repeated source lines from the LLM response, only keep the first occurrence."""
        import re
        # Find all lines that look like 'Answer based on the provided context from ...'
        lines = response.splitlines()
        seen_source = False
        cleaned_lines = []
        for line in lines:
            if re.match(r"^Answer based on the provided context from ", line):
                if not seen_source:
                    cleaned_lines.append(line)
                    seen_source = True
                # else skip repeated source lines
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def xbrl_facts_to_markdown_table(self, xbrl_facts):
        if not xbrl_facts:
            return "No XBRL facts available."
        # Get company and year from the first fact's metadata
        meta0 = xbrl_facts[0]["metadata"]
        company = meta0.get("company_name", "Unknown Company")
        year = meta0.get("year", "Unknown Year")
        filing_date = meta0.get("filing_date", "Unknown Date")
        header_str = f"**Company:** {company}  |  **Year:** {year}  |  **Filing Date:** {filing_date}"
        headers = ["FinancialFact", "Value (USD millions)", "Context"]
        rows = []
        for fact in xbrl_facts:
            meta = fact["metadata"]
            doc = fact["document"]
            element = meta.get("element") or ""
            value = ""
            context = ""
            for line in doc.splitlines():
                if line.startswith("Financial Fact: "):
                    element = line.replace("Financial Fact: ", "").strip()
                elif line.startswith("Value: "):
                    value = line.replace("Value: ", "").strip()
                elif line.startswith("Context: "):
                    context = line.replace("Context: ", "").strip()
            # Convert value to millions if possible
            try:
                value_million = float(value) / 1_000_000
                value = f"{value_million:.2f}"
            except Exception:
                pass
            rows.append([element, value, context])
        # Build Markdown table
        table = header_str + "\n"
        table += "| " + " | ".join(headers) + " |\n"
        table += "|" + "---|" * len(headers) + "\n"
        for row in rows:
            table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        return table

    def xbrl_facts_to_statement_tables(self, xbrl_facts):
        """Group XBRL facts by statement type and create separate markdown tables for each."""
        if not xbrl_facts:
            return "No XBRL facts available."
        
        # Get company and year from the first fact's metadata
        meta0 = xbrl_facts[0]["metadata"]
        company = meta0.get("company_name", "Unknown Company")
        year = meta0.get("year", "Unknown Year")
        filing_date = meta0.get("filing_date", "Unknown Date")
        
        # Group facts by statement type using the new statement_type field
        statement_groups = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow_statement': [],
            'operations_statement': [],
            'comprehensive_income': [],
            'stockholders_equity': []
        }
        
        for fact in xbrl_facts:
            meta = fact["metadata"]
            doc = fact["document"]
            
            # Extract statement type from the document content
            statement_type = 'other'
            for line in doc.splitlines():
                if line.startswith("Context: "):
                    context = line.replace("Context: ", "").strip().lower()
                    # Check for statement type in context
                    if 'income statement' in context or 'operations' in context:
                        statement_type = 'income_statement'
                    elif 'balance sheet' in context:
                        statement_type = 'balance_sheet'
                    elif 'cash flow' in context:
                        statement_type = 'cash_flow_statement'
                    elif 'comprehensive income' in context:
                        statement_type = 'comprehensive_income'
                    elif 'stockholders equity' in context or 'shareholders equity' in context:
                        statement_type = 'stockholders_equity'
                    break
            
            # Also check for statement_type in the fact key (from CSV extraction)
            fact_key = ""
            for line in doc.splitlines():
                if line.startswith("Financial Fact: "):
                    fact_key = line.replace("Financial Fact: ", "").strip()
                    break
            
            # If we can't determine from context, try to infer from fact key
            if statement_type == 'other' and fact_key:
                fact_key_lower = fact_key.lower()
                if any(keyword in fact_key_lower for keyword in ['revenue', 'income', 'profit', 'loss', 'earnings', 'expense']):
                    statement_type = 'income_statement'
                elif any(keyword in fact_key_lower for keyword in ['asset', 'liability', 'equity', 'debt', 'cash']):
                    statement_type = 'balance_sheet'
                elif any(keyword in fact_key_lower for keyword in ['cash', 'flow', 'operating', 'investing', 'financing']):
                    statement_type = 'cash_flow_statement'
            
            if statement_type in statement_groups:
                statement_groups[statement_type].append(fact)
        
        # Create tables for each statement type
        tables = []
        headers = ["FinancialFact", "Value (USD millions)", "Context"]
        
        for statement_type, facts in statement_groups.items():
            if not facts:
                continue
                
            # Format statement type name
            statement_name = statement_type.replace('_', ' ').title()
            
            # Create header for this statement
            header_str = f"**{statement_name} - {company} ({year})** | **Filing Date:** {filing_date}"
            
            # Build table rows
            rows = []
            for fact in facts:
                meta = fact["metadata"]
                doc = fact["document"]
                element = meta.get("element") or ""
                value = ""
                context = ""
                
                for line in doc.splitlines():
                    if line.startswith("Financial Fact: "):
                        element = line.replace("Financial Fact: ", "").strip()
                    elif line.startswith("Value: "):
                        value = line.replace("Value: ", "").strip()
                    elif line.startswith("Context: "):
                        context = line.replace("Context: ", "").strip()
                
                # Convert value to millions if possible
                try:
                    value_million = float(value) / 1_000_000
                    value = f"{value_million:.2f}"
                except Exception:
                    pass
                
                rows.append([element, value, context])
            
            # Build markdown table
            table = header_str + "\n"
            table += "| " + " | ".join(headers) + " |\n"
            table += "|" + "---|" * len(headers) + "\n"
            for row in rows:
                table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            
            tables.append(table)
        
        if not tables:
            return "No financial statement data found in XBRL facts."
        
        return "\n\n".join(tables)

    def get_all_financial_statements_text(self, text_chunks: list) -> str:
        """Extract and format all three financial statements from text chunks."""
        if not text_chunks:
            return ""
        
        # Group chunks by section type
        statement_sections = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow_statement': []
        }
        
        for chunk in text_chunks:
            meta = chunk["metadata"]
            section_type = meta.get("section_type", "").lower()
            
            if section_type in statement_sections:
                statement_sections[section_type].append(chunk)
        
        # Format each statement section
        context_parts = []
        context_parts.append("=== FINANCIAL STATEMENTS (Text) ===")
        
        for statement_type, chunks in statement_sections.items():
            if not chunks:
                continue
            
            # Get metadata from first chunk
            meta = chunks[0]["metadata"]
            company = meta.get("company_name", "Unknown")
            filing_type = meta.get("form", "Unknown")
            date = meta.get("filing_date", "Unknown")
            
            # Format statement type name
            statement_name = statement_type.replace('_', ' ').title()
            
            context_parts.append(f"\n--- {statement_name} - {company} {filing_type} ({date}) ---")
            
            # Combine all chunks for this statement
            statement_content = ""
            for chunk in chunks:
                statement_content += chunk["document"] + "\n\n"
            
            # Truncate if too long (but allow more space for financial statements)
            if len(statement_content) > 3000:
                statement_content = statement_content[:3000] + "..."
            
            context_parts.append(statement_content)
        
        return "\n".join(context_parts)

    def ensure_all_financial_statements_in_context(self, text_chunks: list, xbrl_facts: list, 
                                                  company_ticker: str, year: str) -> str:
        """Ensure all three financial statements are included in the LLM context."""
        context_parts = []
        
        # Add XBRL facts grouped by statement type
        if xbrl_facts:
            context_parts.append("=== STRUCTURED XBRL FACTS (By Statement Type) ===")
            context_parts.append(self.xbrl_facts_to_statement_tables(xbrl_facts))
        
        # Add text-based financial statements
        if text_chunks:
            # Check if we have any financial statement sections
            has_financial_sections = any(
                chunk["metadata"].get("section_type", "").lower() in 
                ['income_statement', 'balance_sheet', 'cash_flow_statement']
                for chunk in text_chunks
            )
            
            if has_financial_sections:
                context_parts.append(self.get_all_financial_statements_text(text_chunks))
            else:
                # Fall back to general text formatting
                context_parts.append(self.format_context_with_sections(text_chunks))
        
        return "\n\n".join(context_parts)

    def is_financial_query(self, user_query: str) -> bool:
        """Check if the user query is requesting financial information."""
        query_lower = user_query.lower()
        
        # Financial keywords that indicate the user wants financial data
        financial_keywords = [
            # Revenue and income related
            'revenue', 'sales', 'income', 'earnings', 'profit', 'loss', 'net income', 'gross profit',
            'operating income', 'ebitda', 'ebit', 'margin', 'profitability',
            
            # Financial statements
            'financial', 'financials', 'financial statement', 'income statement', 'balance sheet', 
            'cash flow', 'cashflow', 'statement of cash flows', 'profit and loss', 'p&l',
            
            # Performance metrics and variations
            'performance', 'perform', 'performed', 'performs', 'outperform', 'outperformed', 'underperform', 'underperformed',
            'operating performance', 'financial performance', 'results', 'quarterly results', 'annual results',
            'earnings report', 'financial report', 'financial results',
            
            # Specific financial metrics
            'assets', 'liabilities', 'equity', 'debt', 'cash', 'cash equivalents', 'inventory',
            'accounts receivable', 'accounts payable', 'working capital', 'capital expenditure',
            'depreciation', 'amortization', 'goodwill', 'intangible assets',
            
            # Ratios and analysis
            'ratio', 'financial ratio', 'debt to equity', 'current ratio', 'quick ratio',
            'return on equity', 'roe', 'return on assets', 'roa', 'return on investment', 'roi',
            'earnings per share', 'eps', 'price to earnings', 'p/e', 'book value',
            
            # Growth and trends
            'growth', 'revenue growth', 'earnings growth', 'financial growth', 'trend',
            'financial trend', 'year over year', 'yoy', 'quarter over quarter', 'qoq',
            
            # Valuation and market
            'valuation', 'market cap', 'market capitalization', 'enterprise value', 'ev',
            'free cash flow', 'fcf', 'dividend', 'dividend yield',
            
            # Business metrics
            'cost of revenue', 'cost of goods sold', 'cogs', 'operating expenses', 'sg&a',
            'research and development', 'r&d', 'capital expenditure', 'capex',
            
            # Item 8 and financial sections
            'item 8', 'financial statements', 'consolidated financial statements',
            'management discussion', 'md&a', 'financial condition', 'financial position'
        ]
        
        # Check if any financial keyword is in the query
        return any(keyword in query_lower for keyword in financial_keywords)

    def build_context(self, user_query, debug=False, debug_lines=None):
        import re
        from config import Config
        item_map = {
            "risk factors": ("Item 1A", "risk factors"),
            "business": ("Item 1", "business"),
            "unresolved staff comments": ("Item 1B", "unresolved staff comments"),
            "properties": ("Item 2", "properties"),
            "legal proceedings": ("Item 3", "legal proceedings"),
            "mine safety": ("Item 4", "mine safety"),
            "market": ("Item 5", "market"),
            "selected financial data": ("Item 6", "selected financial data"),
            "management's discussion": ("Item 7", "management's discussion"),
            "md&a": ("Item 7", "management's discussion"),
            "quantitative and qualitative disclosures": ("Item 7A", "quantitative and qualitative disclosures"),
            "financial statements": ("Item 8", "financial statements"),
            "balance sheet": ("Item 8", "balance sheet"),
            "income statement": ("Item 8", "income statement"),
            "cash flow": ("Item 8", "cash flow statement"),
            "controls and procedures": ("Item 9A", "controls and procedures"),
            "other information": ("Item 9B", "other information"),
            "directors": ("Item 10", "directors"),
            "executive compensation": ("Item 11", "executive compensation"),
            "security ownership": ("Item 12", "security ownership"),
            "certain relationships": ("Item 13", "certain relationships"),
            "principal accounting fees": ("Item 14", "principal accounting fees"),
            "exhibits": ("Item 15", "exhibits"),
        }
        ticker_in_query = None
        query_lower = user_query.lower()
        for company in Config.TOP_COMPANIES:
            ticker = company["ticker"].lower()
            name = company["name"].lower()
            aliases = company.get("aliases", [])
            if (
                ticker in query_lower
                or any(alias in query_lower for alias in aliases)
                or name.split()[0] in query_lower
            ):
                ticker_in_query = company["ticker"]
                break
        # Step 3: Enhanced year detection with fallback to conversation history
        year_matches = re.findall(r"20[0-9]{2}", user_query)
        fiscal_years = list(set(year_matches))  # Remove duplicates and convert to list
        if not fiscal_years:
            _, recent_year = self.extract_context_from_history(user_query)
            if recent_year:
                fiscal_years = [recent_year]
                if debug and debug_lines is not None:
                    debug_lines.append(f"[DEBUG] Using year from conversation history: {recent_year}")
        fiscal_year = fiscal_years[0] if fiscal_years else None
        if debug and debug_lines is not None:
            debug_lines.append(f"[DEBUG] Detected years: {fiscal_years}")
        section_filter = None
        for key, (item_code, item_title) in item_map.items():
            if key in query_lower:
                section_filter = (item_code, item_title)
                break
        is_financial = self.is_financial_query(user_query)
        context_parts = []
        if ticker_in_query and fiscal_years:
            all_xbrl_facts = []
            all_text_chunks = []
            for year in fiscal_years:
                if debug and debug_lines is not None:
                    debug_lines.append(f"[DEBUG] Retrieving content for {ticker_in_query} in {year}")
                year_xbrl_facts = self.get_xbrl_facts(ticker_in_query, year) if is_financial else []
                year_text_chunks = self.get_all_text_chunks(ticker_in_query, year)
                all_xbrl_facts.extend(year_xbrl_facts)
                all_text_chunks.extend(year_text_chunks)
                if debug and debug_lines is not None:
                    debug_lines.append(f"[DEBUG] Retrieved {len(year_text_chunks)} text chunks and {len(year_xbrl_facts)} XBRL facts for {year}")
            if debug and debug_lines is not None:
                debug_lines.append(f"[DEBUG] ticker: {ticker_in_query}, fiscal_year: {fiscal_year}")
                debug_lines.append(f"[DEBUG] text_chunks: {len(all_text_chunks)}, xbrl_facts: {len(all_xbrl_facts)}")
                debug_lines.append(f"[DEBUG] is_financial: {is_financial}")
                debug_lines.append(f"[DEBUG] section_filter: {section_filter}")
            if all_xbrl_facts and debug and debug_lines is not None:
                debug_lines.append("[DEBUG] Sample XBRL fact metadata:")
                for fact in all_xbrl_facts[:3]:
                    meta = fact.get('metadata', {})
                    doc = fact.get('document', '')
                    debug_lines.append(f"  - Element: {meta.get('element', 'N/A')}")
                    debug_lines.append(f"    Context: {meta.get('context', 'N/A')}")
                    debug_lines.append(f"    Statement type: {meta.get('statement_type', 'N/A')}")
                    debug_lines.append(f"    Document preview: {doc[:100]}...")
            if section_filter and all_text_chunks:
                if section_filter[0] == "Item 8":
                    pass
                else:
                    all_text_chunks = self.filter_chunks_by_section(all_text_chunks, section_filter)
                    if debug and debug_lines is not None:
                        debug_lines.append(f"[DEBUG] After section filtering: {len(all_text_chunks)} text chunks")
            if is_financial:
                context = self.ensure_all_financial_statements_in_context(all_text_chunks, all_xbrl_facts, ticker_in_query, ", ".join(fiscal_years))
            else:
                context = self.format_context_with_sections(all_text_chunks)
        else:
            xbrl_facts = []
            if ticker_in_query and fiscal_years and is_financial:
                for year in fiscal_years:
                    year_xbrl_facts = self.get_xbrl_facts(ticker_in_query, year)
                    xbrl_facts.extend(year_xbrl_facts)
            query_results = self.query_vector_db(user_query)
            if ticker_in_query and fiscal_years and is_financial and (xbrl_facts or query_results.get("text_results")):
                all_text_chunks = []
                for year in fiscal_years:
                    year_text_chunks = self.get_all_text_chunks(ticker_in_query, year)
                    all_text_chunks.extend(year_text_chunks)
                context = self.ensure_all_financial_statements_in_context(all_text_chunks, xbrl_facts, ticker_in_query, ", ".join(fiscal_years))
            else:
                if xbrl_facts:
                    context_parts.append("=== STRUCTURED XBRL FACTS (Tabular) ===")
                    context_parts.append(self.xbrl_facts_to_markdown_table(xbrl_facts))
                if query_results.get("text_results") and query_results["text_results"].get("documents"):
                    context_parts.append("=== RELEVANT FILING SECTIONS ===")
                    for i, doc in enumerate(query_results["text_results"]["documents"][0]):
                        metadata = query_results["text_results"]["metadatas"][0][i]
                        company = metadata.get("company_name", "Unknown")
                        filing_type = metadata.get("form", "Unknown")
                        date = metadata.get("filing_date", "Unknown")
                        chunk_type = metadata.get("chunk_type", "text")
                        section_name = metadata.get("section_name", metadata.get("item_title", "Unknown Section"))
                        context_parts.append(f"\n--- {company} {filing_type} ({date}) - {section_name} [{chunk_type}] ---")
                        context_parts.append(doc[:1000] + "..." if len(doc) > 1000 else doc)
                context = "\n".join(context_parts)
        if debug and debug_lines is not None:
            debug_lines.append(f"[DEBUG] Final context length: {len(context)}")
            debug_lines.append(f"[DEBUG] Context preview:\n{context[:1000]}\n{'...' if len(context) > 1000 else ''}")
        return context

    def generate_response(self, user_query: str) -> str:
        """
        Enhanced to handle follow-up questions with context from conversation history.
        Uses the build_context method for consistent context building.
        """
        import re
        
        # Step 1: Enhance query with context from conversation history
        enhanced_query = self.enhance_query_with_context(user_query)
        print(f"Original query: '{user_query}'")
        print(f"Enhanced query: '{enhanced_query}'")
        
        # Use the build_context method for consistent context building
        context = self.build_context(enhanced_query)
        
        print(f"\n===== CONTEXT SENT TO LLM =====\nCharacters: {len(context)}\nPreview:\n{context[:1000]}\n...\n===============================\n")
        
        # Build the full prompt (simpler approach like our debug version)
        # Check if this is a follow-up question that needs conversation context
        query_lower = user_query.lower()
        pronouns = ["its", "their", "the company", "this company", "the firm"]
        temporal_refs = ["same year", "last year", "this year", "that year", "the year"]
        is_followup = any(pronoun in query_lower for pronoun in pronouns) or any(ref in query_lower for ref in temporal_refs)
        
        if is_followup and self.chat_history:
            # Include recent conversation history for context
            recent_history = self.chat_history[-4:] if len(self.chat_history) >= 4 else self.chat_history
            conversation_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])
            
            full_prompt = f"""You are a financial analyst assistant. Use the following SEC filing information to answer the user's question.

Previous conversation context:
{conversation_context}

Context from SEC filings:
{context}

User Question: {user_query}

Please provide a clear, direct answer to the user's question in a conversational and systematic tone.\n- Format your answer using Markdown: use **bold** for key numbers, bullet points for lists, and tables for structured data.\n- Add line breaks for readability.\n- Cite your sources naturally within the answer or at the end, mentioning the company, filing type, date, and section as appropriate.\n- Do not use numbered headings or explicit section labels—keep the response flowing and natural.\nIf the information is not available in the provided context, say so clearly."""
        else:
            full_prompt = f"""You are a financial analyst assistant. Use the following SEC filing information to answer the user's question.

Context from SEC filings:
{context}

User Question: {user_query}

Please provide a clear, direct answer to the user's question in a conversational and systematic tone.\n- Format your answer using Markdown: use **bold** for key numbers, bullet points for lists, and tables for structured data.\n- Add line breaks for readability.\n- Cite your sources naturally within the answer or at the end, mentioning the company, filing type, date, and section as appropriate.\n- Do not use numbered headings or explicit section labels—keep the response flowing and natural.\nIf the information is not available in the provided context, say so clearly."""
        
        # Use the same API call structure as our working debug version
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": Config.DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    llm_response = response_data['choices'][0]['message']['content']
                    # Post-process to remove repeated source lines
                    llm_response = self.postprocess_llm_response(llm_response)
                    self.chat_history.append({"role": "user", "content": user_query})
                    self.chat_history.append({"role": "assistant", "content": llm_response})
                    if len(self.chat_history) > 20:
                        self.chat_history = self.chat_history[-20:]
                    return llm_response
                else:
                    return f"OpenRouter API error: Unexpected response format - {response_data}"
            else:
                return f"OpenRouter API error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"API call failed: {str(e)}"

    def get_available_companies(self) -> List[str]:
        """Get list of companies available in the database."""
        try:
            companies = set()
            if self.collection.count() > 0:
                results = self.collection.get(include=["metadatas"])
                for metadata in results["metadatas"]:
                    if metadata and "company_name" in metadata:
                        companies.add(metadata["company_name"])
            return sorted(list(companies))
        except Exception as e:
            st.error(f"Error getting available companies: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        try:
            total_count = self.collection.count()
            return {
                "total_text_documents": total_count,
                "total_documents": total_count,
                "available_companies": self.get_available_companies()
            }
        except Exception as e:
            st.error(f"Error getting database stats: {e}")
            return {} 

    def extract_context_from_history(self, user_query: str) -> tuple:
        """
        Extract company and year context from conversation history.
        Returns (ticker, year) tuple or (None, None) if no context found.
        """
        if not self.chat_history:
            return None, None
        
        # Look for the most recent company and year mentioned
        recent_ticker = None
        recent_year = None
        
        # Search through recent history (last 4 messages)
        recent_history = self.chat_history[-4:] if len(self.chat_history) >= 4 else self.chat_history
        
        for msg in reversed(recent_history):  # Start from most recent
            if msg["role"] == "user":
                # Extract ticker from user message
                query_lower = msg["content"].lower()
                for company in Config.TOP_COMPANIES:
                    ticker = company["ticker"].lower()
                    name = company["name"].lower()
                    aliases = company.get("aliases", [])
                    if (
                        ticker in query_lower
                        or any(alias in query_lower for alias in aliases)
                        or name.split()[0] in query_lower
                    ):
                        recent_ticker = company["ticker"]
                        break
                
                # Extract year from user message
                import re
                year_match = re.search(r"20[0-9]{2}", msg["content"])
                if year_match:
                    recent_year = year_match.group(0)
                
                if recent_ticker and recent_year:
                    break
        
        return recent_ticker, recent_year

    def enhance_query_with_context(self, user_query: str) -> str:
        """
        Enhance user query by adding context from conversation history.
        Handles pronouns and temporal references.
        """
        query_lower = user_query.lower()
        
        # Check for pronouns that need context
        pronouns = ["its", "their", "the company", "this company", "the firm"]
        has_pronoun = any(pronoun in query_lower for pronoun in pronouns)
        
        # Check for temporal references
        temporal_refs = ["same year", "last year", "this year", "that year", "the year"]
        has_temporal_ref = any(ref in query_lower for ref in temporal_refs)
        
        if has_pronoun or has_temporal_ref:
            recent_ticker, recent_year = self.extract_context_from_history(user_query)
            
            if recent_ticker and recent_year:
                # Replace pronouns with company name
                for company in Config.TOP_COMPANIES:
                    if company["ticker"] == recent_ticker:
                        company_name = company["name"]
                        enhanced_query = user_query
                        enhanced_query = enhanced_query.replace("its", f"{company_name}'s")
                        enhanced_query = enhanced_query.replace("their", f"{company_name}'s")
                        enhanced_query = enhanced_query.replace("the company", company_name)
                        enhanced_query = enhanced_query.replace("this company", company_name)
                        enhanced_query = enhanced_query.replace("the firm", company_name)
                        break
                
                # Replace temporal references with specific year
                enhanced_query = enhanced_query.replace("same year", f"{recent_year}")
                enhanced_query = enhanced_query.replace("last year", f"{recent_year}")
                enhanced_query = enhanced_query.replace("this year", f"{recent_year}")
                enhanced_query = enhanced_query.replace("that year", f"{recent_year}")
                enhanced_query = enhanced_query.replace("the year", f"{recent_year}")
                
                return enhanced_query
        
        return user_query 