import os
import json
import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from retriever import SECVectorStore
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBIngestion:
    def __init__(self):
        self.config = Config()
        from extractor import SECDataCollector
        self.data_collector = SECDataCollector()
        self.vector_store = SECVectorStore()
        
    def load_metadata(self) -> Dict[str, Any]:
        """Load the filings metadata"""
        metadata_file = os.path.join(self.config.DATA_DIR, 'filings_metadata.json')
        
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found: {metadata_file}")
            logger.info("Please run extractor.py first to download the filings.")
            return {}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_text_filing(self, ticker: str, accession_number: str) -> str:
        """Load text filing from file"""
        filename = f"{ticker}_{accession_number}.txt"
        filepath = os.path.join(self.config.FILINGS_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Text filing not found: {filepath}")
            return ""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_xbrl_csv_data(self, ticker: str, accession_number: str) -> Dict[str, pd.DataFrame]:
        """Load XBRL CSV files for a filing"""
        xbrl_dir = os.path.join(self.config.XBRL_DIR, f"{ticker}_{accession_number}")
        
        if not os.path.exists(xbrl_dir):
            logger.warning(f"XBRL directory not found: {xbrl_dir}")
            return {}
        
        # Load all CSV files in the directory
        csv_data = {}
        for filename in os.listdir(xbrl_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(xbrl_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    # Extract statement type from filename (e.g., balance_sheet, income_statement)
                    statement_type = filename.replace(f"{ticker}_{accession_number}_", "").replace(".csv", "")
                    csv_data[statement_type] = df
                    logger.info(f"Loaded {statement_type} CSV: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Could not read CSV file {filename}: {str(e)}")
        
        return csv_data
    
    def extract_financial_facts_from_csv(self, ticker: str, accession_number: str) -> Dict[str, Any]:
        """Extract financial facts from XBRL CSV files"""
        csv_data = self.load_xbrl_csv_data(ticker, accession_number)
        
        if not csv_data:
            return {}
        
        facts = {
            'filing_info': {
            'accession_number': accession_number,
            'ticker': ticker,
                'extraction_date': datetime.now().isoformat(),
                'statements_available': list(csv_data.keys())
            },
            'financial_facts': {},
            'statement_summaries': {}
        }
        
        # Find the most recent fiscal year across all statements
        all_years = set()
        for statement_type, df in csv_data.items():
            for col in df.columns:
                if isinstance(col, str) and len(col) >= 4 and col[:4].isdigit():
                    year = col[:4]
                    if year.isdigit():
                        all_years.add(int(year))
        fiscal_year = str(max(all_years)) if all_years else None
        
        # Process each statement type
        for statement_type, df in csv_data.items():
            if df.empty:
                continue
            
            # Create summary for this statement
            statement_summary = {
                'rows': len(df),
                'columns': list(df.columns),
                'fiscal_year': fiscal_year,
                'data_types': df.dtypes.to_dict()
            }
            facts['statement_summaries'][statement_type] = statement_summary
            
            # Extract key financial facts from the DataFrame
            financial_facts = {}
            
            # Look for common financial metrics in the DataFrame
            for idx, row in df.iterrows():
                # Get the concept/label (usually first column)
                concept = str(row.iloc[0]) if len(row) > 0 else ""
                label = str(row.iloc[1]) if len(row) > 1 else concept
                
                # Skip abstract/concept rows that don't have actual values
                if concept.lower().endswith('abstract') or concept.lower().endswith('concept'):
                    continue
                
                # Get the value for the most recent fiscal year column
                value = None
                if fiscal_year:
                    for col in df.columns:
                        if col.startswith(fiscal_year):
                            try:
                                val = row[col]
                                if pd.notna(val) and str(val).strip() and str(val).lower() not in ['nan', 'false', 'true']:
                                    float_val = float(val)
                                    if float_val != 0:
                                        value = str(val)
                                        break
                            except (ValueError, TypeError, KeyError):
                                continue
                
                if value and concept and concept != "nan" and not concept.lower().endswith('abstract'):
                    fact_key = f"{statement_type}_{concept}_{idx}"
                            financial_facts[fact_key] = {
                        'element': concept,
                        'label': label,
                        'value': value,
                        'statement_type': statement_type,
                        'context': f"{statement_type} statement",
                        'fiscal_year': fiscal_year
                    }
            
            facts['financial_facts'].update(financial_facts)
        
        # Set fiscal year if found
        if fiscal_year:
            facts['fiscal_year'] = fiscal_year
        
        logger.info(f"Extracted {len(facts['financial_facts'])} financial facts from {ticker}_{accession_number}")
        
        return facts
    
    def ingest_all_data(self, clear_collection=True) -> Dict[str, Any]:
        """Ingest all filing data into vector database. If clear_collection is True, clears the vector store first."""
        logger.info("Starting vector database ingestion...")
        
        # Load metadata
        metadata = self.load_metadata()
        if not metadata:
            return {}
        
        if clear_collection:
        logger.info("Clearing existing vector database collection...")
        self.vector_store.clear_collection()
        
        # Track ingestion statistics
        stats = {
            'total_filings': 0,
            'text_chunks_added': 0,
            'financial_facts_added': 0,
            'companies_processed': 0,
            'errors': []
        }
        
        # Process each company
        for ticker, company_data in metadata['companies'].items():
            logger.info(f"Processing company: {ticker}")
            stats['companies_processed'] += 1
            
            for filing in company_data['filings']:
                stats['total_filings'] += 1
                accession_number = filing['accession_number']
                try:
                    # Try to extract fiscal_year from XBRL CSV data
                    fiscal_year = None
                    if filing.get('xbrl_statements_saved'):
                        facts = self.extract_financial_facts_from_csv(ticker, accession_number)
                        if facts and 'fiscal_year' in facts:
                            fiscal_year = facts['fiscal_year']
                    # Fallback: use filing_date if fiscal_year is still None
                    if not fiscal_year and filing.get('filing_date'):
                        fiscal_year = filing['filing_date'][:4]
                    filing_with_fiscal = dict(filing)
                    if fiscal_year:
                        filing_with_fiscal['fiscal_year'] = fiscal_year
                    # Ingest text filing
                    if filing.get('text_downloaded', False):
                        text_content = self.load_text_filing(ticker, accession_number)
                        if text_content:
                            chunks_added = self.vector_store.add_filing_text(filing_with_fiscal, text_content)
                            stats['text_chunks_added'] += chunks_added
                    # Ingest XBRL data from CSV files
                    if filing.get('xbrl_statements_saved'):
                        facts = self.extract_financial_facts_from_csv(ticker, accession_number)
                            if facts:
                            if 'fiscal_year' not in filing_with_fiscal and 'fiscal_year' in facts:
                                filing_with_fiscal['fiscal_year'] = facts['fiscal_year']
                            facts_added = self.vector_store.add_financial_facts(filing_with_fiscal, facts)
                                stats['financial_facts_added'] += facts_added
                except Exception as e:
                    error_msg = f"Error processing filing {accession_number} for {ticker}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
        # Get final statistics
        final_stats = self.vector_store.get_statistics()
        stats.update(final_stats)
        logger.info("Vector database ingestion completed!")
        logger.info(f"Statistics: {stats}")
        return stats
    
    def verify_ingestion(self) -> Dict[str, Any]:
        logger.info("Verifying vector database ingestion...")
        stats = self.vector_store.get_statistics()
        test_queries = [
            "revenue",
            "risk factors",
            "management discussion",
            "financial statements"
        ]
        search_results = {}
        for query in test_queries:
            results = self.vector_store.search(query, n_results=3)
            search_results[query] = len(results)
        verification = {
            'vector_db_stats': stats,
            'search_test_results': search_results,
            'verification_passed': stats.get('total_documents', 0) > 0
        }
        logger.info(f"Verification results: {verification}")
        return verification

# Main entry point for full ingestion
if __name__ == "__main__":
    print("Ingesting data with fixed XBRL extraction logic (ALL COMPANIES)...")
    print("=" * 60)
    ingestion = VectorDBIngestion()
    # Check if data exists
    metadata_file = os.path.join(Config.DATA_DIR, 'filings_metadata.json')
    if not os.path.exists(metadata_file):
        print("No metadata file found. Please run extractor.py first.")
        exit(1)
    # Ingest all data (clearing collection first)
    stats = ingestion.ingest_all_data(clear_collection=True)
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY (ALL COMPANIES)")
    print("=" * 60)
        print(f"Companies processed: {stats.get('companies_processed', 0)}")
        print(f"Total filings: {stats.get('total_filings', 0)}")
        print(f"Text chunks added: {stats.get('text_chunks_added', 0)}")
        print(f"Financial facts added: {stats.get('financial_facts_added', 0)}")
        print(f"Total documents in vector DB: {stats.get('total_documents', 0)}")
        if stats.get('errors'):
            print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
                print(f"  - {error}")
    # Verify ingestion
    print("\nVerifying ingestion...")
    verification = ingestion.verify_ingestion()
    print(f"Verification passed: {verification.get('verification_passed', False)}")
    print("\nIngestion for ALL COMPANIES completed!") 