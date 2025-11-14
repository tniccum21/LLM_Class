# Week08 - Amazon Product Search Agent

## Project Overview
Intelligent Amazon product search agent with weighted scoring and LLM-based relevance filtering.

## Requirements Specification

### Core Features
1. **Amazon Search**: SerpAPI amazon_search endpoint
2. **Multi-Page Results**: 2 pages per search (configurable)
3. **Weighted Scoring**: `score = rating × log10(review_count + 1)`
4. **LLM Relevance Filter**: Batch processing with gpt-oss-20b via LM Studio
5. **CLI Interface**: argparse with multiple configuration options

### Technical Stack
- **Search API**: SerpAPI (250 free searches, amazon.com marketplace)
- **LLM**: gpt-oss-20b via LM Studio (local inference at http://192.168.7.171:1234)
- **Language**: Python 3.11+
- **Interface**: Command-line (argparse)
- **Dependencies**: serpapi, python-dotenv, requests

### Configuration Parameters

**CLI Arguments** (Option B - More Control):
```bash
python amazon_search.py "Organic White Beans" \
  --top 10 \
  --pages 2 \
  --min-reviews 10 \
  --batch-size 10
```

**Required**:
- `query`: Search term (positional argument)

**Optional**:
- `--top N`: Number of results to display (default: 10)
- `--pages N`: Pages to fetch from SerpAPI (default: 2)
- `--min-reviews N`: Minimum review count filter (default: 10)
- `--batch-size N`: LLM batch size for relevance checking (default: 10)
- `--cache`: Enable result caching to conserve SerpAPI quota

### Data Flow Architecture

```
[User Input] → [SerpAPI Search] → [Extract Data] → [Score Calculation]
                                                            ↓
[Display Results] ← [Sort & Limit] ← [Relevance Filter (LLM Batch)]
```

**Step-by-Step Process**:
1. Parse CLI arguments and validate configuration
2. Query SerpAPI with amazon_search (2 pages = ~40-60 products)
3. Extract: title, rating, review_count, price, url, asin
4. Calculate scores: `rating × log10(review_count + 1)`
5. Filter: reviews >= min_reviews threshold
6. Batch LLM relevance check (batches of 10 products)
7. Sort by score (descending)
8. Display top N results in detailed table format

### LLM Integration Details

**Relevance Filtering**: Binary classification (Option A)
- Prompt: "Is this product relevant to '{query}'? Answer YES or NO."
- Batch processing: 10 products per LLM call
- Response parsing: Extract YES/NO for each product
- Filtering: Keep only YES products

**Batch Strategy**:
- 60 products → 6 batches of 10
- Adjustable via `--batch-size` parameter
- Prevents token limit issues
- Faster than one-by-one evaluation

### Scoring Formula

**Logarithmic Weighting** (base 10):
```python
score = rating × log10(review_count + 1)
```

**Examples**:
- 4.8★ with 10,000 reviews: 4.8 × 4.0 = 19.2
- 4.5★ with 10,000 reviews: 4.5 × 4.0 = 18.0
- 4.8★ with 100 reviews: 4.8 × 2.0 = 9.6
- 4.5★ with 1,000 reviews: 4.5 × 3.0 = 13.5

### Output Format

**Detailed Table** (Option B):
```
Amazon Product Search: "Organic White Beans"
Found 45 relevant products (from 58 total results)
Showing top 10 by weighted score

Rank | Product                           | Score | Rating | Reviews | Price  | URL
-----|-----------------------------------|-------|--------|---------|--------|----
1    | Brand X Organic White Beans 6pk   | 19.2  | 4.8★   | 10,234  | $24.99 | amzn.to/xyz
2    | Brand Y Organic Cannellini Beans  | 18.5  | 4.7★   | 8,521   | $18.99 | amzn.to/abc
```

### File Structure

```
Week08/
├── amazon_search.py          # Main CLI application
├── serpapi_client.py         # SerpAPI integration
├── llm_filter.py             # LLM relevance filtering
├── scorer.py                 # Scoring algorithm
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── secrets.env               # API keys (gitignored)
├── README.md                 # User guide
└── DESIGN.md                 # Technical specification
```

### Environment Variables

**secrets.env**:
```bash
SERPAPI_API_KEY=your_key_here
LM_STUDIO_URL=http://192.168.7.171:1234
LLM_MODEL=gpt-oss-20b
```

## Implementation Phases

### Phase 1: Core MVP
- SerpAPI integration with amazon_search
- Basic scoring algorithm
- Simple CLI with --query and --top
- Table output format

### Phase 2: LLM Integration
- LM Studio connection
- Batch relevance filtering
- Filter low-review products

### Phase 3: Polish & Optimization
- Result caching
- Error handling
- Configuration validation
- Documentation
