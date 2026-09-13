import os
import json
import re
from typing import Dict, Any, List

from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket


SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ kỹ thuật khách hàng của Vingroup.
- Giọng điệu: Chuyên nghiệp, thân thiện, chính xác và đáng tin cậy.

## 2. AVAILABLE TOOLS
Bạn có quyền truy cập vào các công cụ sau:
{tools}

## 3. CORE RULES
- KHÔNG BAO GIỜ tự bịa đặt (hallucinate) thông tin, giá cả, thông số sản phẩm hoặc trạng thái đơn hàng.
- BẮT BUỘC phải gọi tool để lấy dữ liệu thực tế trước khi trả lời các câu hỏi về sản phẩm, dịch vụ hoặc xử lý yêu cầu hỗ trợ.
- Chỉ trả lời dựa trên dữ liệu thực tế nhận được từ Observation. Nếu tool báo lỗi hoặc không tìm thấy kết quả, hãy thông báo trung thực với khách hàng.

## 4. OPERATIONAL BOUNDARIES
- Phạm vi hoạt động: Chỉ giải đáp và hỗ trợ các vấn đề liên quan đến hệ sinh thái Vingroup (VinFast, Vinhomes, Vinmec, Vinpearl, v.v.).
- Nếu người dùng hỏi các chủ đề ngoài phạm vi (ví dụ: xe của hãng khác, chính trị, v.v.), hãy từ chối lịch sự và cho biết bạn chỉ hỗ trợ dịch vụ của Vingroup.

## 5. OUTPUT CONTRACT (ReAct Format)
Bạn BẮT BUỘC phải suy nghĩ và hành động theo định dạng chính xác sau đây. Vòng lặp Thought/Action/Action Input/Observation có thể lặp lại nhiều lần cho đến khi bạn có đủ thông tin.

Thought: Suy nghĩ của bạn về những gì người dùng đang hỏi và bạn cần làm gì tiếp theo.
Action: Tên của công cụ cần gọi (chọn 1 trong các tool được cung cấp). Nếu không cần tool hoặc đã đủ thông tin, ghi "None".
Action Input: Tham số truyền vào tool dưới dạng JSON hợp lệ (ví dụ: {{"category": "xe_dien", "max_price": 600000000}}).
Observation: Kết quả trả về từ hệ thống (BẠN KHÔNG ĐƯỢC TỰ VIẾT PHẦN NÀY, hệ thống sẽ điền).
... (lặp lại Thought/Action/Action Input/Observation nếu cần)
Thought: Tôi đã có đủ thông tin để trả lời.
Final Answer: Câu trả lời cuối cùng gửi đến người dùng bằng ngôn ngữ tự nhiên, mạch lạc.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME")
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API Key)."""
        if self.model:
            try:
                prompt = f"Bạn là trợ lý AI. Hãy trả lời câu hỏi sau: {user_input}"
                response = self.model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                answer = f"[Lỗi gọi API]: {str(e)}"
        else:
            answer = f"[Chatbot Baseline Mock] Trả lời ảo cho: {user_input} (Cần cấu hình GEMINI_API_KEY)"

        return {
            "answer": answer,
            "tool_calls": [],
            "status": "success",
            "mode": "baseline"
        }

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling (ReAct Loop)."""

    def __init__(self, max_iterations: int = 5, api_key: str = None, model_name: str = None):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME")

        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy ReAct Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})
        
        if not self.model:
            return {
                "answer": "[Lỗi] Không tìm thấy GEMINI_API_KEY. Vui lòng cấu hình biến môi trường.",
                "trace": self.trace,
                "iterations": 0,
                "status": "error"
            }
            
        tools_str = json.dumps(TOOL_DEFINITIONS, indent=2, ensure_ascii=False)

        current_prompt = SYSTEM_PROMPT.format(tools=tools_str) + f"\nUser: {user_input}\n"
        
        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.model.generate_content(current_prompt)
                llm_output = response.text
                self.trace.append({"step": f"llm_output_iter_{iteration}", "content": llm_output})
                
                current_prompt += llm_output + "\n"

                final_answer_match = re.search(r"Final Answer:\s*(.*)", llm_output, re.DOTALL)
                if final_answer_match:
                    return {
                        "answer": final_answer_match.group(1).strip(),
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "success"
                    }

                action_match = re.search(r"Action:\s*(.+)", llm_output)
                action_input_match = re.search(r"Action Input:\s*(.+)", llm_output, re.DOTALL)

                if action_match and action_input_match:
                    action_name = action_match.group(1).strip()
                    action_input_str = action_input_match.group(1).strip()
                    
                    action_input_str = re.sub(r"^```json|```$", "", action_input_str).strip()
                    
                    try:
                        action_args = json.loads(action_input_str)
                    except json.JSONDecodeError:
                        observation = "Error: Action Input không phải là JSON hợp lệ."
                        current_prompt += f"Observation: {observation}\n"
                        self.trace.append({"step": f"tool_error_iter_{iteration}", "error": observation})
                        continue

                    self.trace.append({"step": f"tool_execution_iter_{iteration}", "tool": action_name, "args": action_args})
                    
                    if action_name in TOOL_MAP:
                        tool_result = TOOL_MAP[action_name](**action_args)
                        observation = str(tool_result)
                    else:
                        observation = f"Error: Tool '{action_name}' không tồn tại."

                    current_prompt += f"Observation: {observation}\n"
                    self.trace.append({"step": f"observation_iter_{iteration}", "content": observation})
                else:
                    current_prompt += "Observation: Format không hợp lệ. Vui lòng sử dụng đúng định dạng Thought/Action/Action Input hoặc Final Answer.\n"

            except Exception as e:
                self.trace.append({"step": f"error_iter_{iteration}", "error": str(e)})
                break

        return {
            "answer": "Không thể tìm ra câu trả lời sau số lần lặp tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    baseline_result = chatbot.query(user_query)
    print("Baseline Answer:\n", baseline_result["answer"])
    print("-" * 50)

    print("\n=== RUNNING TOOL CALLING AGENT (ReAct) ===")
    agent = ToolCallingAgent(max_iterations=5)
    agent_result = agent.run(user_query)
    print("Agent Final Answer:\n", agent_result["answer"])
    
    print("\n=== TRACE LOG ===")
    print(json.dumps(agent_result["trace"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()