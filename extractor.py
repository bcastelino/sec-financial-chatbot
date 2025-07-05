import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from edgar import *
from config import Config
import logging
import requests
import re

# Configure logging to suppress verbose httpx and other library logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("edgar").setLevel(logging.WARNING)

class SECDataCollector:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        set_identity(self.config.EDGAR_IDENTITY)
        
    def get_all_filings(self, filing_type: str = "10-K") -> List[Dict]:
        try:
            logger.info(f"Getting all {filing_type} filings for all companies")
            years = range(self.config.START_YEAR, self.config.END_YEAR + 1)
            all_filings = get_filings(form=[filing_type], year=years)
            logger.info(f"Found {len(all_filings)} total {filing_type} filings in the specified range")
            all_company_filings = []
            for company in self.config.TOP_COMPANIES:
                ticker = company['ticker']
                logger.info(f"Filtering filings for {ticker}")
                ticker_filings = all_filings.filter(ticker=[ticker])
                if not ticker_filings:
                    logger.warning(f"No {filing_type} filings found for {ticker} in the specified years")
                    continue
                logger.info(f"Found {len(ticker_filings)} relevant filings for {ticker}")
                for filing in ticker_filings:
                    filing_date = filing.filing_date
                    if isinstance(filing_date, str):
                        filing_date = datetime.strptime(filing_date, '%Y-%m-%d').date()
                    elif hasattr(filing_date, 'date'):
                        filing_date = filing_date.date()
                    company_name = next((c['name'] for c in self.config.TOP_COMPANIES if c['ticker'] == ticker), ticker)
                    all_company_filings.append({
                        'filing_obj': filing,
                        'accession_number': filing.accession_number,
                        'filing_date': filing_date.isoformat() if hasattr(filing_date, 'isoformat') else str(filing_date),
                        'form': filing.form,
                        'company_name': company_name,
                        'ticker': ticker,
                        'file_number': filing.file_number if hasattr(filing, 'file_number') else None
                    })
            logger.info(f"Total processed filings: {len(all_company_filings)}")
            return all_company_filings
        except Exception as e:
            logger.error(f"Error getting all filings: {str(e)}")
            return []
    
    def get_company_filings(self, ticker: str, filing_type: str = "10-K", years_back: int = None) -> List[Dict]:
        try:
            logger.info(f"Getting {filing_type} filings for {ticker}")
            all_filings = self.get_all_filings(filing_type)
            company_filings = [f for f in all_filings if f['ticker'] == ticker]
            company_filings = company_filings[:5]
            logger.info(f"Found {len(company_filings)} {filing_type} filings for {ticker}")
            return company_filings
        except Exception as e:
            logger.error(f"Error getting filings for {ticker}: {str(e)}")
            return []
    
    def get_all_company_filings(self, ticker: str) -> List[Dict]:
        years_back = self.config.END_YEAR - self.config.START_YEAR + 1
        filings = self.get_company_filings(ticker, "10-K", years_back=years_back)
        logger.info(f"Total 10-K filings found for {ticker}: {len(filings)}")
        return filings
    
    def download_filing_text(self, filing_obj, ticker: str) -> Optional[str]:
        try:
            tenk = filing_obj.obj()
            content_to_save = []
            if tenk.business:
                content_to_save.append("--- BUSINESS ---\n")
                content_to_save.append(tenk.business.strip())
                content_to_save.append("\n\n")
            if tenk.risk_factors:
                content_to_save.append("--- RISK FACTORS ---\n")
                content_to_save.append(tenk.risk_factors.strip())
                content_to_save.append("\n\n")
            if tenk.management_discussion:
                content_to_save.append("--- MANAGEMENT'S DISCUSSION AND ANALYSIS ---\n")
                content_to_save.append(tenk.management_discussion.strip())
                content_to_save.append("\n\n")
            if tenk.balance_sheet:
                content_to_save.append("--- BALANCE SHEET ---\n")
                content_to_save.append(str(tenk.balance_sheet).strip())
                content_to_save.append("\n\n")
            if tenk.income_statement:
                content_to_save.append("--- INCOME STATEMENT ---\n")
                content_to_save.append(str(tenk.income_statement).strip())
                content_to_save.append("\n\n")
            if tenk.cash_flow_statement:
                content_to_save.append("--- CASH FLOW STATEMENT ---\n")
                content_to_save.append(str(tenk.cash_flow_statement).strip())
                content_to_save.append("\n\n")
            if not content_to_save:
                logger.warning(f"No structured sections found for {filing_obj.accession_number}, using raw text")
                text_content = filing_obj.text()
                content_to_save.append(text_content)
            filename = f"{ticker}_{filing_obj.accession_number}.txt"
            filepath = os.path.join(self.config.FILINGS_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(content_to_save)
            logger.info(f"Downloaded text filing: {filename}")
            return ''.join(content_to_save)
        except Exception as e:
            logger.error(f"Error downloading text filing {filing_obj.accession_number}: {str(e)}")
            return None
    
    def extract_and_save_xbrl_statements(self, filing_obj, ticker: str, accession_number: str, filing_date: str) -> Dict[str, str]:
        """Extract XBRL statements as DataFrames and save as CSV in data/xbrl. Returns dict of saved file paths."""
        saved_files = {}
        try:
            # Access XBRL directly from filing object, not from TenK object
            xbrl_obj = filing_obj.xbrl()
            statement_map = {
                'balance_sheet': 'CONSOLIDATEDBALANCESHEETS',
                'income_statement': 'CONSOLIDATEDSTATEMENTSINCOME',
                'operations_statement': 'CONSOLIDATEDSTATEMENTSOFOPERATIONS',
                'cash_flow_statement': 'CONSOLIDATEDSTATEMENTSCASHFLOWS',
                'comprehensive_income': 'CONSOLIDATEDSTATEMENTSOFCOMPREHENSIVEINCOME',
                'stockholders_equity': 'CONSOLIDATEDSTATEMENTSOFSTOCKHOLDERSEQUITY'
            }
            for attr, statement_name in statement_map.items():
                try:
                    df = xbrl_obj.statements.to_dataframe(statement_name)
                    if df is not None and not df.empty:
                        xbrl_dir = os.path.join(self.config.XBRL_DIR, f"{ticker}_{accession_number}")
                        os.makedirs(xbrl_dir, exist_ok=True)
                        csv_filename = f"{ticker}_{accession_number}_{attr}.csv"
                        csv_path = os.path.join(xbrl_dir, csv_filename)
                        df.to_csv(csv_path, index=False)
                        saved_files[attr] = csv_path
                        logger.info(f"Saved {attr} as CSV: {csv_path}")
                    else:
                        logger.info(f"No data for {attr} in {ticker} {accession_number}")
                except Exception as e:
                    logger.warning(f"Error extracting {attr} for {ticker} {accession_number}: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting XBRL statements for {ticker} {accession_number}: {str(e)}")
        return saved_files
    
    def collect_all_data(self) -> Dict[str, Any]:
        all_data = {
            'companies': {},
            'filings_metadata': [],
            'xbrl_files': {}
        }
        all_filings = self.get_all_filings("10-K")
        filings_by_ticker = {}
        for filing in all_filings:
            ticker = filing['ticker']
            if ticker not in filings_by_ticker:
                filings_by_ticker[ticker] = []
            filings_by_ticker[ticker].append(filing)
        for company in self.config.TOP_COMPANIES:
            ticker = company['ticker']
            logger.info(f"Processing company: {ticker}")
            company_filings = filings_by_ticker.get(ticker, [])
            company_data = {
                'name': company['name'],
                'ticker': ticker,
                'filings': []
            }
            for filing in company_filings:
                accession_number = filing['accession_number']
                filing_obj = filing['filing_obj']
                filing_date = filing['filing_date']
                # Download text filing
                text_content = self.download_filing_text(filing_obj, ticker)
                # Extract and save XBRL statements as CSV
                xbrl_saved = self.extract_and_save_xbrl_statements(filing_obj, ticker, accession_number, filing_date)
                    filing_metadata = {
                    'accession_number': accession_number,
                    'filing_date': filing_date,
                        'form': filing['form'],
                        'company_name': filing['company_name'],
                        'ticker': filing['ticker'],
                        'file_number': filing['file_number'],
                        'text_downloaded': text_content is not None,
                    'xbrl_statements_saved': list(xbrl_saved.keys())
                    }
                    company_data['filings'].append(filing_metadata)
                    all_data['filings_metadata'].append(filing_metadata)
                if xbrl_saved:
                    all_data['xbrl_files'][accession_number] = xbrl_saved
            all_data['companies'][ticker] = company_data
        metadata_file = os.path.join(self.config.DATA_DIR, 'filings_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, default=str)
        logger.info(f"Data collection completed. Metadata saved to {metadata_file}")
        return all_data
    
    def get_filing_summary(self) -> pd.DataFrame:
        metadata_file = os.path.join(self.config.DATA_DIR, 'filings_metadata.json')
        if not os.path.exists(metadata_file):
            logger.warning("No metadata file found. Run collect_all_data() first.")
            return pd.DataFrame()
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary_data = []
        for filing in data['filings_metadata']:
            summary_data.append({
                'Company': filing['company_name'],
                'Ticker': filing['ticker'],
                'Form': filing['form'],
                'Filing Date': filing['filing_date'],
                'Text Downloaded': filing['text_downloaded'],
                'XBRL Statements Saved': ', '.join(filing.get('xbrl_statements_saved', []))
            })
        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    collector = SECDataCollector()
    print("Starting 10-K data collection for all companies...")
    all_data = collector.collect_all_data()
    with open(os.path.join(collector.config.DATA_DIR, 'filings_metadata.json'), 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nData collection completed!")
    print(f"Total companies processed: {len(all_data['companies'])}")
    print(f"Total filings processed: {len(all_data['filings_metadata'])}")
    print(f"Filings with XBRL statements saved: {sum(1 for v in all_data['xbrl_files'].values() if v)}")
    for ticker, company_data in all_data['companies'].items():
        print(f"{ticker}: {len(company_data['filings'])} filings") 