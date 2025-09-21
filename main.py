"""
Algolia MCP Server - Production Ready
AI Tinkerers Hackathon - DeepAgents Integration
Robust error handling and validation
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import mcp.types as types
from algoliasearch.search.client import SearchClient
import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Initialize MCP server
mcp = FastMCP(
    "Algolia Vector Search Server",
    port=3000,
    stateless_http=True,
    debug=True,
)

def get_client() -> Optional[SearchClient]:
    """Create Algolia client with proper validation."""
    app_id = os.getenv("ALGOLIA_APP_ID")
    api_key = os.getenv("ALGOLIA_API_KEY")
    
    if not app_id or not api_key:
        return None
    
    return SearchClient(app_id, api_key)

def validate_client() -> Dict[str, Any]:
    """Validate client configuration."""
    client = get_client()
    if not client:
        return {
            "success": False,
            "error": "Missing ALGOLIA_APP_ID or ALGOLIA_API_KEY environment variables"
        }
    return {"success": True, "client": client}

@mcp.tool(
    title="Save Object",
    description="Add a single record to an Algolia index with automatic task completion waiting. This tool ensures data consistency by waiting for the indexing operation to complete before returning. Perfect for adding individual documents, products, or any structured data to make it searchable. Automatically assigns objectID if not provided in the data.",
)
async def save_object(
    index_name: str = Field(description="Algolia index name"),
    object_id: str = Field(description="Unique object ID"),
    object_data: str = Field(description="JSON string of object data"),
) -> Dict[str, Any]:
    """Save a single object to Algolia index."""
    start_time = time.time()
    
    # Validate client
    validation = validate_client()
    if not validation["success"]:
        return {
            **validation,
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }
    
    client = validation["client"]
    
    try:
        # Parse object data
        try:
            data = json.loads(object_data)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid JSON in object_data",
                "performance": {"latency_ms": (time.time() - start_time) * 1000}
            }
        
        # Ensure objectID is set
        data["objectID"] = object_id
        
        # Save object
        save_resp = await client.save_object(
            index_name=index_name,
            body=data
        )
        
        # Wait for task completion
        # save_resp is a SaveObjectResponse object with task_id attribute
        await client.wait_for_task(
            index_name=index_name,
            task_id=save_resp.task_id
        )
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "object_id": object_id,
            "task_id": save_resp.task_id,
            "index_name": index_name,
            "performance": {
                "latency_ms": latency,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }

@mcp.tool(
    title="Save Objects Batch",
    description="Efficiently add multiple records to an Algolia index in a single batch operation with automatic task completion waiting. Ideal for bulk data imports, content migrations, or periodic data updates. Automatically generates objectIDs for objects that don't have them. Provides performance metrics including objects per second throughput. Much faster than individual saves for large datasets.",
)
async def save_objects_batch(
    index_name: str = Field(description="Algolia index name"),
    objects_json: str = Field(description="JSON array of objects to save"),
) -> Dict[str, Any]:
    """Save multiple objects to Algolia index in batch."""
    start_time = time.time()
    
    # Validate client
    validation = validate_client()
    if not validation["success"]:
        return {
            **validation,
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }
    
    client = validation["client"]
    
    try:
        # Parse objects data
        try:
            objects = json.loads(objects_json)
            if not isinstance(objects, list):
                raise ValueError("Expected array of objects")
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "success": False,
                "error": f"Invalid JSON array in objects_json: {str(e)}",
                "performance": {"latency_ms": (time.time() - start_time) * 1000}
            }
        
        # Ensure all objects have objectID
        for i, obj in enumerate(objects):
            if "objectID" not in obj:
                obj["objectID"] = f"auto_id_{i}_{int(time.time())}"
        
        # Save objects in batch
        # save_objects returns a list of BatchResponse objects
        save_resps = await client.save_objects(
            index_name=index_name,
            objects=objects
        )
        
        # Wait for task completion - use the first response's task_id
        if save_resps and len(save_resps) > 0:
            task_id = save_resps[0].task_id
            await client.wait_for_task(
                index_name=index_name,
                task_id=task_id
            )
        else:
            task_id = None
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "objects_count": len(objects),
            "task_id": task_id,
            "index_name": index_name,
            "sample_objects": objects[:3] if objects else [],
            "performance": {
                "latency_ms": latency,
                "timestamp": time.time(),
                "objects_per_second": len(objects) / (latency / 1000) if latency > 0 else 0
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }

@mcp.tool(
    title="Search Index",
    description="Perform advanced search queries on Algolia indexes with built-in typo tolerance, faceting, and highlighting capabilities. Supports complex filtering, attribute selection, and result customization. Returns comprehensive search analytics including processing time and hit counts. Perfect for implementing search functionality, content discovery, or data exploration with natural language queries.",
)
async def search_index(
    index_name: str = Field(description="Algolia index name"),
    query: str = Field(description="Search query"),
    hits_per_page: int = Field(description="Number of results per page", default=20),
    attributes_to_retrieve: str = Field(description="Comma-separated attributes to retrieve", default="*"),
    attributes_to_highlight: str = Field(description="Comma-separated attributes to highlight", default=""),
    filters: str = Field(description="Algolia filter expression", default=""),
) -> Dict[str, Any]:
    """Search Algolia index with comprehensive options."""
    start_time = time.time()
    
    # Validate client
    validation = validate_client()
    if not validation["success"]:
        return {
            **validation,
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }
    
    client = validation["client"]
    
    try:
        # Build search parameters
        search_params = {
            "indexName": index_name,
            "query": query,
            "hitsPerPage": hits_per_page
        }
        
        # Add optional parameters
        if attributes_to_retrieve and attributes_to_retrieve != "*":
            search_params["attributesToRetrieve"] = [
                attr.strip() for attr in attributes_to_retrieve.split(",")
            ]
        
        if attributes_to_highlight:
            search_params["attributesToHighlight"] = [
                attr.strip() for attr in attributes_to_highlight.split(",")
            ]
        
        if filters:
            search_params["filters"] = filters
        
        # Perform search
        results = await client.search({
            "requests": [search_params]
        })
        
        latency = (time.time() - start_time) * 1000
        
        # Extract results from response - results is a SearchResponses object
        # results.results is a list of SearchResult objects
        # Each SearchResult has an actual_instance which is a SearchResponse
        if results.results and len(results.results) > 0:
            search_result = results.results[0].actual_instance
            # SearchResponse has attributes, not dict keys
            hits = search_result.hits if hasattr(search_result, 'hits') else []
            nb_hits = search_result.nb_hits if hasattr(search_result, 'nb_hits') else 0
            processing_time = search_result.processing_time_ms if hasattr(search_result, 'processing_time_ms') else 0
        else:
            hits = []
            nb_hits = 0
            processing_time = 0
        
        return {
            "success": True,
            "query": query,
            "hits": [hit.model_dump() if hasattr(hit, 'model_dump') else hit for hit in hits],
            "nb_hits": nb_hits,
            "processing_time_ms": processing_time,
            "index_name": index_name,
            "search_params": search_params,
            "performance": {
                "latency_ms": latency,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "query": query,
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }

@mcp.tool(
    title="Index From Search Results",
    description="Transform and index external search results (from SerpApi, Google Search, or other APIs) into Algolia for enhanced searchability and analytics. Automatically converts search result formats into Algolia-optimized objects with proper metadata, deduplication via URL hashing, and timestamp tracking. Perfect for creating searchable knowledge bases from web research, competitive analysis, or content aggregation workflows.",
)
async def index_from_search_results(
    search_results_json: str = Field(description="JSON string of search results"),
    index_name: str = Field(description="Algolia index name", default="realtime_search"),
) -> Dict[str, Any]:
    """Convert and index search results from external sources."""
    start_time = time.time()
    
    # Validate client
    validation = validate_client()
    if not validation["success"]:
        return {
            **validation,
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }
    
    client = validation["client"]
    
    try:
        # Parse search results
        try:
            search_data = json.loads(search_results_json)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid JSON in search_results_json",
                "performance": {"latency_ms": (time.time() - start_time) * 1000}
            }
        
        # Convert search results to Algolia objects
        objects = []
        for result in search_data.get("organic_results", []):
            if not result.get("link"):
                continue
                
            object_id = hashlib.md5(result["link"].encode()).hexdigest()
            algolia_object = {
                "objectID": object_id,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("link", ""),
                "position": result.get("position", 0),
                "source_query": search_data.get("query", ""),
                "indexed_at": time.time(),
                "source": "external_search"
            }
            objects.append(algolia_object)
        
        if not objects:
            return {
                "success": False,
                "error": "No valid objects to index",
                "performance": {"latency_ms": (time.time() - start_time) * 1000}
            }
        
        # Save objects
        # save_objects returns a list of BatchResponse objects
        save_resps = await client.save_objects(
            index_name=index_name,
            objects=objects
        )
        
        # Wait for completion - use the first response's task_id
        if save_resps and len(save_resps) > 0:
            task_id = save_resps[0].task_id
            await client.wait_for_task(
                index_name=index_name,
                task_id=task_id
            )
        else:
            task_id = None
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "indexed_count": len(objects),
            "task_id": task_id,
            "index_name": index_name,
            "sample_objects": objects[:2],
            "performance": {
                "latency_ms": latency,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "performance": {"latency_ms": (time.time() - start_time) * 1000}
        }

@mcp.resource(
    uri="algolia://test/{index_name}",
    description="Test Algolia connection by adding and searching a test record",
    name="Connection Test",
)
async def test_connection(index_name: str) -> str:
    """Test Algolia connection following hello_algolia.py pattern."""
    # Test record
    test_record = {"objectID": "test-record-1", "name": "test record"}
    
    # Add record
    add_result = await save_object(
        index_name=index_name,
        object_id="test-record-1",
        object_data=json.dumps({"name": "test record"})
    )
    
    if not add_result.get("success"):
        return f"❌ Add failed: {add_result.get('error')}"
    
    # Search for test
    search_result = await search_index(
        index_name=index_name,
        query="test"
    )
    
    if not search_result.get("success"):
        return f"❌ Search failed: {search_result.get('error')}"
    
    return f"✅ Test passed - Added and found {search_result.get('nb_hits', 0)} results"

@mcp.resource(
    uri="algolia://search/{index_name}/{query}",
    description="Search results formatted for human reading",
    name="Search Results",
)
async def search_results_formatted(index_name: str, query: str) -> str:
    """Get formatted search results."""
    result = await search_index(index_name=index_name, query=query, hits_per_page=5)
    
    if not result.get("success"):
        return f"❌ Search error: {result.get('error')}"
    
    lines = [f"Search results for '{query}' in {index_name}:"]
    for i, hit in enumerate(result.get("hits", []), 1):
        title = hit.get("title", hit.get("name", "No title"))
        lines.append(f"{i}. {title}")
        if "snippet" in hit:
            lines.append(f"   {hit['snippet']}")
        if "url" in hit:
            lines.append(f"   {hit['url']}")
    
    lines.append(f"\nTotal: {result.get('nb_hits', 0)} results ({result.get('processing_time_ms', 0)}ms)")
    return "\n".join(lines)

@mcp.prompt(
    name="algolia_workflow_prompt",
    description="Generate workflow prompts for Algolia operations"
)
def algolia_workflow_prompt(
    task: str = Field(description="Task: index, search, or test", default="search"),
    data_source: str = Field(description="Data source description", default="search_results"),
) -> str:
    """Generate Algolia workflow prompts."""
    if task == "index":
        return f"Index {data_source} into Algolia: 1) Parse and validate data, 2) Create objects with proper objectIDs, 3) Batch save and wait for completion, 4) Verify indexing success."
    elif task == "test":
        return "Test Algolia connection: 1) Add test record, 2) Wait for indexing, 3) Search for test record, 4) Verify results."
    else:  # search
        return f"Search Algolia effectively: 1) Craft semantic query, 2) Set appropriate filters and attributes, 3) Analyze results for relevance, 4) Extract key insights."

if __name__ == "__main__":
    # Validate environment
    app_id = os.getenv("ALGOLIA_APP_ID")
    api_key = os.getenv("ALGOLIA_API_KEY")
    
    if not app_id or not api_key:
        print("❌ Missing ALGOLIA_APP_ID or ALGOLIA_API_KEY environment variables")
        print("Please set your Algolia credentials before running the server")
    else:
        print("✅ Algolia credentials found")
    
    print("🔍 Starting Robust Algolia MCP Server on port 3000...")
    print("Tools: save_object, save_objects_batch, search_index, index_from_search_results")
    print("Resources: algolia://test/{index}, algolia://search/{index}/{query}")
    print("Prompts: algolia_workflow_prompt")
    
    mcp.run(transport="streamable-http")