import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

def search_product_catalog(category: str, max_price: int = 999_999_999_999) -> List[Dict[str, Any]]:
    """Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa."""
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    try:
        with open(catalog_file, "r", encoding="utf-8") as file:
            products = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        return [{"error": f"Unable to read product catalog: {error}"}]

    if not isinstance(products, list):
        return [{"error": "Product catalog must contain a JSON list."}]

    return [
        product
        for product in products
        if isinstance(product, dict)
        and product.get("category") == category
        and isinstance(product.get("price_vnd"), (int, float))
        and product["price_vnd"] <= max_price
    ]


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium",
) -> Dict[str, Any]:
    """Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket."""
    if priority not in {"low", "medium", "high"}:
        return {"error": "Priority must be one of: low, medium, high."}

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    try:
        if os.path.exists(tickets_file):
            with open(tickets_file, "r", encoding="utf-8") as file:
                tickets = json.load(file)
        else:
            tickets = []
    except (OSError, json.JSONDecodeError) as error:
        return {"error": f"Unable to read support tickets: {error}"}

    if not isinstance(tickets, list):
        return {"error": "Support tickets must contain a JSON list."}

    now = datetime.now()
    today = now.strftime("%Y%m%d")
    prefix = f"TK-{today}-"
    existing_sequences = [
        int(ticket["ticket_id"][len(prefix):])
        for ticket in tickets
        if isinstance(ticket, dict)
        and isinstance(ticket.get("ticket_id"), str)
        and ticket["ticket_id"].startswith(prefix)
        and ticket["ticket_id"][len(prefix):].isdigit()
    ]
    ticket = {
        "ticket_id": f"{prefix}{max(existing_sequences, default=0) + 1:03d}",
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(timespec="seconds"),
    }
    tickets.append(ticket)

    try:
        with open(tickets_file, "w", encoding="utf-8") as file:
            json.dump(tickets, file, ensure_ascii=False, indent=2)
    except OSError as error:
        return {"error": f"Unable to save support ticket: {error}"}
    return ticket


TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm hoặc dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["xe_dien", "du_lich"], "description": "Danh mục sản phẩm cần tra cứu."},
                "max_price": {"type": "integer", "minimum": 0, "description": "Giá tối đa tính bằng VNĐ."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Tạo ticket hỗ trợ khách hàng mới.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Tên khách hàng."},
                "issue_description": {"type": "string", "description": "Mô tả vấn đề cần hỗ trợ."},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium", "description": "Mức độ ưu tiên của ticket."},
            },
            "required": ["customer_name", "issue_description"],
        },
    }
]


TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
