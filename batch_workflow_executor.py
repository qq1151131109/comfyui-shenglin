import json
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
import os

logger = logging.getLogger(__name__)

class BatchWorkflowExecutorNode:
    """
    批量工作流执行器节点
    支持将工作流分发到多个ComfyUI实例执行
    """

    def __init__(self):
        self.session = None
        self.execution_results = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {"multiline": True, "tooltip": "ComfyUI API工作流JSON"}),
                "comfyui_instances": ("STRING", {"multiline": True, "tooltip": "ComfyUI实例地址列表，每行一个，可来自上一个节点"}),
                "prompt_list": ("STRING", {"multiline": True, "tooltip": "提示词列表，每行一个"}),
            },
            "optional": {
                "placeholder": ("STRING", {"default": "{{PROMPT}}", "tooltip": "主要占位符，将被提示词列表替换"}),
                "replacement_config": ("STRING", {"multiline": True, "default": "", "tooltip": "额外替换配置，格式: placeholder=value，每行一个"}),
                "timeout_seconds": ("INT", {"default": 300, "min": 60, "max": 1800, "tooltip": "单个任务超时时间（秒）"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("execution_summary", "results_json")
    FUNCTION = "batch_execute"
    CATEGORY = "🔥 Shenglin/工作流"

    def parse_instances(self, instances_text: str) -> List[str]:
        """解析ComfyUI实例地址列表"""
        instances = []
        for line in instances_text.strip().split('\n'):
            line = line.strip()
            if line:
                # 确保地址格式正确
                if not line.startswith('http'):
                    line = f"http://{line}"
                if not line.endswith('/'):
                    line += '/'
                instances.append(line)
        return instances

    def parse_prompts(self, prompts_text: str) -> List[str]:
        """解析提示词列表"""
        prompts = []
        for line in prompts_text.strip().split('\n'):
            line = line.strip()
            if line:
                prompts.append(line)
        return prompts

    def parse_replacement_config(self, config_text: str) -> Dict[str, str]:
        """解析额外替换配置"""
        replacements = {}
        if not config_text.strip():
            return replacements

        for line in config_text.strip().split('\n'):
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                replacements[key.strip()] = value.strip()

        return replacements

    def replace_workflow_placeholders(self, workflow: Dict, placeholder: str, value: str,
                                    extra_replacements: Dict[str, str] = None) -> Dict:
        """在整个工作流JSON中替换所有占位符"""
        # 将工作流转换为字符串进行全局替换
        workflow_str = json.dumps(workflow, ensure_ascii=False)

        # 替换主要占位符
        main_count = workflow_str.count(placeholder)
        if main_count > 0:
            workflow_str = workflow_str.replace(placeholder, value)
            logger.info(f"替换主要占位符 '{placeholder}' 为 '{value[:50]}...' (共{main_count}处)")
        else:
            logger.warning(f"工作流中未找到主要占位符 '{placeholder}'")

        # 替换额外占位符
        if extra_replacements:
            for old_placeholder, new_value in extra_replacements.items():
                count = workflow_str.count(old_placeholder)
                if count > 0:
                    workflow_str = workflow_str.replace(old_placeholder, new_value)
                    logger.info(f"替换额外占位符 '{old_placeholder}' 为 '{new_value[:30]}...' (共{count}处)")

        # 转换回字典
        workflow_copy = json.loads(workflow_str)
        return workflow_copy

    async def check_instance_health(self, session: aiohttp.ClientSession, instance_url: str) -> bool:
        """检查ComfyUI实例健康状态"""
        try:
            async with session.get(f"{instance_url}system_stats", timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"实例 {instance_url} 健康检查失败: {e}")
            return False

    async def execute_workflow_on_instance(self, session: aiohttp.ClientSession,
                                         instance_url: str, workflow: Dict,
                                         prompt_id: str, timeout_seconds: int) -> Dict:
        """在指定实例上执行工作流"""
        try:
            # 提交工作流
            payload = {"prompt": workflow, "client_id": prompt_id}
            async with session.post(f"{instance_url}prompt",
                                  json=payload,
                                  timeout=30) as response:
                if response.status != 200:
                    return {
                        "success": False,
                        "error": f"提交失败，状态码: {response.status}",
                        "instance": instance_url
                    }

                result = await response.json()
                prompt_id_server = result.get("prompt_id")

            # 轮询执行状态
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                async with session.get(f"{instance_url}history/{prompt_id_server}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id_server in history:
                            execution_data = history[prompt_id_server]
                            if execution_data.get("status", {}).get("completed", False):
                                return {
                                    "success": True,
                                    "prompt_id": prompt_id_server,
                                    "instance": instance_url,
                                    "execution_time": time.time() - start_time,
                                    "outputs": execution_data.get("outputs", {})
                                }
                            elif "status" in execution_data and "error" in execution_data["status"]:
                                return {
                                    "success": False,
                                    "error": execution_data["status"]["error"],
                                    "instance": instance_url
                                }

                await asyncio.sleep(2)  # 等待2秒后再检查

            return {
                "success": False,
                "error": f"执行超时（{timeout_seconds}秒）",
                "instance": instance_url
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "instance": instance_url
            }

    async def execute_single_task(self, session: aiohttp.ClientSession,
                                available_instances: List[str], workflow: Dict,
                                prompt: str, task_id: int, placeholder: str,
                                extra_replacements: Dict[str, str], timeout_seconds: int,
                                retry_count: int, enable_retry: bool) -> Dict:
        """执行单个任务"""
        # 替换工作流中的占位符
        modified_workflow = self.replace_workflow_placeholders(workflow, placeholder, prompt, extra_replacements)

        # 尝试在可用实例上执行
        for attempt in range(retry_count + 1 if enable_retry else 1):
            for instance_url in available_instances:
                prompt_id = f"batch_{task_id}_{attempt}_{uuid.uuid4().hex[:8]}"

                logger.info(f"任务 {task_id} 尝试 {attempt + 1} - 在实例 {instance_url} 上执行")
                result = await self.execute_workflow_on_instance(
                    session, instance_url, modified_workflow, prompt_id, timeout_seconds
                )

                if result["success"]:
                    result.update({
                        "task_id": task_id,
                        "prompt": prompt,
                        "attempt": attempt + 1
                    })
                    return result
                else:
                    logger.warning(f"任务 {task_id} 在 {instance_url} 执行失败: {result['error']}")

            if attempt < retry_count and enable_retry:
                logger.info(f"任务 {task_id} 等待 5 秒后重试...")
                await asyncio.sleep(5)

        return {
            "success": False,
            "task_id": task_id,
            "prompt": prompt,
            "error": "所有实例和重试都失败了",
            "attempt": retry_count + 1 if enable_retry else 1
        }

    def batch_execute(self, workflow_json: str, comfyui_instances: str, prompt_list: str,
                     placeholder: str = "{{PROMPT}}", replacement_config: str = "",
                     timeout_seconds: int = 300) -> Tuple[str, str]:
        """批量执行工作流"""

        try:
            # 解析输入
            workflow = json.loads(workflow_json)
            instances = self.parse_instances(comfyui_instances)
            prompts = self.parse_prompts(prompt_list)
            extra_replacements = self.parse_replacement_config(replacement_config)

            if not instances:
                return ("错误: 没有提供有效的ComfyUI实例", "{}")

            if not prompts:
                return ("错误: 没有提供有效的提示词", "{}")

            logger.info(f"开始批量执行: {len(prompts)} 个任务，{len(instances)} 个实例")
            if extra_replacements:
                logger.info(f"额外替换配置: {list(extra_replacements.keys())}")

            # 异步执行批量任务
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # 内部使用固定的重试设置，后台自动重试
                enable_retry = True
                retry_count = 2
                # 根据服务器数量自动决定并发数，避免过载单个服务器
                max_concurrent = len(instances)
                results = loop.run_until_complete(
                    self._async_batch_execute(workflow, instances, prompts, placeholder,
                                            extra_replacements, max_concurrent, timeout_seconds,
                                            enable_retry, retry_count)
                )
            finally:
                loop.close()

            # 生成执行摘要
            successful_tasks = [r for r in results if r["success"]]
            failed_tasks = [r for r in results if not r["success"]]

            total_time = sum([r.get("execution_time", 0) for r in successful_tasks])
            avg_time = total_time / len(successful_tasks) if successful_tasks else 0

            summary = f"""🚀 批量执行完成

📊 执行统计:
  - 总任务数: {len(prompts)}
  - 成功: {len(successful_tasks)}
  - 失败: {len(failed_tasks)}
  - 成功率: {len(successful_tasks)/len(prompts)*100:.1f}%

⏱️ 时间统计:
  - 总执行时间: {total_time:.1f}秒
  - 平均执行时间: {avg_time:.1f}秒/任务

🖥️ 实例使用:
  - 可用实例数: {len(instances)}
  - 最大并发数: {max_concurrent}
"""

            if failed_tasks:
                summary += f"\n❌ 失败任务:\n"
                for task in failed_tasks[:5]:  # 只显示前5个失败任务
                    summary += f"  - 任务 {task['task_id']}: {task['error']}\n"
                if len(failed_tasks) > 5:
                    summary += f"  - ... 还有 {len(failed_tasks) - 5} 个失败任务\n"

            # 返回结果
            results_json = json.dumps(results, ensure_ascii=False, indent=2)
            return (summary, results_json)

        except Exception as e:
            logger.error(f"批量执行失败: {e}")
            return (f"执行失败: {str(e)}", "{}")

    async def _async_batch_execute(self, workflow: Dict, instances: List[str],
                                 prompts: List[str], placeholder: str,
                                 extra_replacements: Dict[str, str], max_concurrent: int,
                                 timeout_seconds: int, enable_retry: bool, retry_count: int) -> List[Dict]:
        """异步批量执行"""

        async with aiohttp.ClientSession() as session:
            # 检查实例健康状态
            logger.info("检查ComfyUI实例健康状态...")
            health_checks = await asyncio.gather(*[
                self.check_instance_health(session, instance)
                for instance in instances
            ])

            available_instances = [
                instances[i] for i, healthy in enumerate(health_checks) if healthy
            ]

            if not available_instances:
                raise Exception("没有可用的ComfyUI实例")

            logger.info(f"可用实例: {len(available_instances)}/{len(instances)}")

            # 创建任务
            semaphore = asyncio.Semaphore(max_concurrent)

            async def execute_with_semaphore(task_id, prompt):
                async with semaphore:
                    return await self.execute_single_task(
                        session, available_instances, workflow, prompt, task_id,
                        placeholder, extra_replacements, timeout_seconds, retry_count, enable_retry
                    )

            # 并发执行所有任务
            tasks = [
                execute_with_semaphore(i, prompt)
                for i, prompt in enumerate(prompts)
            ]

            results = await asyncio.gather(*tasks)
            return results


# 节点注册
NODE_CLASS_MAPPINGS = {
    "BatchWorkflowExecutorNode": BatchWorkflowExecutorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchWorkflowExecutorNode": "批量工作流执行器"
}