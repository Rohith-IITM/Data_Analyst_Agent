import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import time
import logging
from urllib.parse import urlparse
import re
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_url(url: str) -> Dict[str, Any]:
    """Validate URL format and return error dict if invalid, None if valid"""
    try:
        if not isinstance(url, str):
            return {
                'success': False,
                'error': f'URL must be a string, got {type(url).__name__}',
                'url': str(url),
                'tables': []
            }
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return {
                'success': False,
                'error': f'Invalid URL format: {url}',
                'url': url,
                'tables': []
            }
        return None  # Valid URL
    except Exception as e:
        return {
            'success': False,
            'error': f'URL validation failed: {str(e)}',
            'url': str(url),
            'tables': []
        }


class WebTableScrapingTool:
    """
    Web table scraping tool optimized for LangGraph AI agent systems.
    This tool provides a clean interface for agents to extract table data
    from web pages with comprehensive error handling and structured output.
    """

    def __init__(self,
                 delay: float = 1.0,
                 timeout: int = 10,
                 max_retries: int = 3,
                 user_agent: str = None):
        """
        Initialize the web table scraping tool.
        Args:
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            user_agent: Custom user agent string
        """
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent or 'Mozilla/5.0 (compatible; AI-Agent-TableScraper/1.0)'
        })

    def extract_tables(self, url: str,
                      table_selector: Optional[str] = None,
                      include_metadata: bool = True) -> Dict[str, Any]:
        """
        Main tool function to extract tables from a webpage.
        Args:
            url: The URL to scrape
            table_selector: Optional CSS selector for specific tables
            include_metadata: Whether to include detailed metadata
        Returns:
            Structured dictionary with extraction results
        """
        # Validate URL first
        validation_error = validate_url(url)
        if validation_error:
            return validation_error

        logger.info(f"Starting table extraction from: {url}")
        for attempt in range(self.max_retries):
            try:
                # Fetch webpage content
                response = self._fetch_webpage(url)
                if not response['success']:
                    if attempt == self.max_retries - 1:
                        return {
                            'success': False,
                            'error': response['error'],
                            'url': url,
                            'tables': []
                        }
                    continue

                # Parse HTML and extract tables
                soup = BeautifulSoup(response['content'], 'html.parser')
                tables = self._find_tables(soup, table_selector)

                if not tables:
                    return {
                        'success': True,
                        'url': url,
                        'tables_found': 0,
                        'tables': [],
                        'message': 'No tables found on this webpage'
                    }

                # Extract data from all tables
                extracted_tables = []
                for i, table in enumerate(tables):
                    table_data = self._manual_table_processing(table, i, include_metadata)
                    extracted_tables.append(table_data)

                result = {
                    'success': True,
                    'url': url,
                    'tables_found': len(extracted_tables),
                    'tables': extracted_tables
                }

                if include_metadata:
                    result['extraction_metadata'] = {
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'total_rows': sum(t.get('row_count', 0) for t in extracted_tables),
                        'total_columns': sum(t.get('column_count', 0) for t in extracted_tables)
                    }

                logger.info(f"Successfully extracted {len(extracted_tables)} tables from {url}")
                return result

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        'success': False,
                        'error': f'Table extraction failed after {self.max_retries} attempts: {str(e)}',
                        'url': url,
                        'tables': []
                    }
                time.sleep(self.delay * (attempt + 1))  # Exponential backoff

    def _fetch_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch webpage content with error handling"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return {
                'success': True,
                'content': response.content,
                'encoding': response.encoding,
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Failed to fetch webpage: {str(e)}'
            }

    def _find_tables(self, soup: BeautifulSoup, selector: Optional[str] = None) -> List:
        """Find tables using CSS selector or default search, excluding nested tables."""
        if selector:
            tables = soup.select(selector)
        else:
            tables = soup.find_all('table')

        # Remove nested tables (those inside another <table>)
        filtered_tables = []
        table_set = {id(t) for t in tables}
        for t in tables:
            parent = t.find_parent('table')
            if parent is None or id(parent) not in table_set:
                filtered_tables.append(t)

        logger.info(f"Found {len(tables)} tables, {len(filtered_tables)} after removing nested ones.")
        return filtered_tables

    def _extract_headers(self, table, table_rows: List) -> List[str]:
        """Extract table headers only if they are meaningful."""
        headers = []

        # Check thead
        thead = table.find('thead')
        if thead:
            header_cells = thead.find_all(['th', 'td'])
            headers = [self._clean_cell_text(cell) for cell in header_cells]
            if any(h.strip() for h in headers):
                # Avoid purely numeric or placeholder headers
                if not all(h.isdigit() for h in headers if h.strip()):
                    return headers

        # Check first row for th elements
        if table_rows:
            first_row_cells = table_rows[0].find_all(['th', 'td'])
            if any(cell.name == 'th' for cell in first_row_cells):
                candidate_headers = [self._clean_cell_text(cell) for cell in first_row_cells]
                if any(h.strip() for h in candidate_headers):
                    if not all(h.isdigit() for h in candidate_headers if h.strip()):
                        return candidate_headers

        return []  # No valid headers found

    def _manual_table_processing(self, table, index: int, include_metadata: bool) -> Dict[str, Any]:
        """Manual table processing — now the only method used."""
        rows_data = []
        table_rows = table.find_all('tr')

        if not table_rows:
            result = {
                'table_id': f'table_{index}',
                'row_count': 0,
                'column_count': 0,
                'data': [],
                'columns': [],
                'extraction_method': 'manual'
            }
            if include_metadata:
                result.update(self._extract_table_metadata(table))
            return result

        # Extract headers
        headers = self._extract_headers(table, table_rows)
        use_headers = bool(headers)
        expected_col_count = len(headers) if headers else None

        # Determine data rows
        data_rows = table_rows[1:] if use_headers else table_rows

        # Track seen rows to avoid duplicates (e.g., header repeated as data)
        seen_rows = set()

        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            row_text = tuple(self._clean_cell_text(cell) for cell in cells)

            # Skip empty rows
            if not any(t.strip() for t in row_text):
                continue

            # Skip if this row matches the header values exactly
            if row_text == tuple(headers):
                continue

            # Skip duplicate rows
            if row_text in seen_rows:
                continue
            seen_rows.add(row_text)

            # Trim or pad to expected column count
            row_list = list(row_text)
            if expected_col_count:
                if len(row_list) > expected_col_count:
                    row_list = row_list[:expected_col_count]
                elif len(row_list) < expected_col_count:
                    row_list += [''] * (expected_col_count - len(row_list))

            # Build dict
            if use_headers:
                row_dict = dict(zip(headers, row_list))
            else:
                row_dict = {f'col_{i}': val for i, val in enumerate(row_list)}

            rows_data.append(row_dict)

        # Final column list
        columns = headers or ([f'col_{i}' for i in range(expected_col_count)] if expected_col_count else [])

        result = {
            'table_id': f'table_{index}',
            'row_count': len(rows_data),
            'column_count': len(columns),
            'data': rows_data,
            'columns': columns,
            'extraction_method': 'manual'
        }

        if include_metadata:
            result.update(self._extract_table_metadata(table))

        return result

    def _extract_table_metadata(self, table) -> Dict[str, Any]:
        """Extract additional metadata from table element"""
        metadata = {}
        if table.attrs:
            metadata['html_attributes'] = dict(table.attrs)
        caption = table.find('caption')
        if caption:
            metadata['caption'] = self._clean_cell_text(caption)
        if 'summary' in table.attrs:
            metadata['summary'] = table.attrs['summary']
        return metadata

    def _clean_cell_text(self, cell) -> str:
        """Clean and normalize cell text content"""
        if not cell:
            return ''
        text = cell.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())  # Normalize whitespace
        return text


# Tool functions for LangGraph integration
def scrape_web_tables(url: str,
                     table_selector: Optional[str] = None,
                     include_metadata: bool = True,
                     delay: float = 1.0) -> str:
    """
    LangGraph tool function to scrape tables from a webpage.
    Args:
        url: The webpage URL to scrape
        table_selector: Optional CSS selector for specific tables
        include_metadata: Include detailed metadata in results
        delay: Delay between requests (seconds)
    Returns:
        JSON string with extracted table data
    """
    tool = WebTableScrapingTool(delay=delay)
    result = tool.extract_tables(url, table_selector, include_metadata)
    clean_result = {
        'success': result.get('success', False),
        'url': result.get('url', url),
        'tables_found': result.get('tables_found', 0),
        'tables': result.get('tables', [])
    }
    if 'message' in result:
        clean_result['message'] = result['message']
    if 'error' in result:
        clean_result['error'] = result['error']
    if 'extraction_metadata' in result:
        clean_result['extraction_metadata'] = result['extraction_metadata']
    return json.dumps(clean_result, indent=2, ensure_ascii=False, default=str)

@tool
def web_table_summarizer_tool(url: Annotated[str, "URL of the website where the data is present"]) -> str:
    """
    LangGraph tool function to get a quick summary of tables on a webpage.
    Args:
        url: The webpage URL to analyze
    Returns:
        JSON string with table summary information including sample rows
    """
    tool = WebTableScrapingTool(delay=0.5)
    result = tool.extract_tables(url, include_metadata=False)
    if not result.get('success', False):
        clean_error = {
            'success': False,
            'url': url,
            'error': result.get('error', 'Unknown error occurred'),
            'tables_found': 0
        }
        return json.dumps(clean_error, indent=2, default=str)

    summary = {
        'success': True,
        'url': url,
        'tables_found': result.get('tables_found', 0),
        'summary': []
    }
    for table in result.get('tables', []):
        # Get up to first 2 sample rows
        sample_rows = table.get('data', [])[:2]

        # Clean sample rows to avoid very long values
        cleaned_sample_rows = []
        for row in sample_rows:
            cleaned_row = {}
            for k, v in row.items():
                key = str(k)[:50] + "..." if len(str(k)) > 50 else str(k)
                val = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                cleaned_row[key] = val
            cleaned_sample_rows.append(cleaned_row)

        summary['summary'].append({
            'table_id': table.get('table_id', 'unknown'),
            'rows': table.get('row_count', 0),
            'columns': table.get('column_count', 0),
            'column_names': table.get('columns', [])[:10],  # First 10 column names
            'sample_rows': cleaned_sample_rows  # First 2 rows of data
        })
    return str(json.dumps(summary, indent=2, default=str))
