from typing import Any, List
import hashlib
import io
import base64
from pathlib import Path
from pydantic import Field
from pydantic.dataclasses import dataclass
from fpdf import FPDF

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.api import logger

from .tag import (
    build_detail_message,
    FilterConfig,
    filter_illusts_with_reason,
    process_and_send_illusts_sorted,
)
from .pixiv_utils import (
    send_pixiv_image,
    send_forward_message,
    generate_safe_filename,
)


@dataclass
class PixivIllustSearchTool(FunctionTool[AstrAgentContext]):
    """
    Pixiv插画搜索工具
    """

    pixiv_client: Any = None
    pixiv_config: Any = None
    pixiv_client_wrapper: Any = None
    name: str = "pixiv_search_illust"
    description: str = (
        "【图片/插画搜索专用工具】用于在Pixiv上搜索二次元插画、动漫图片、壁纸等。"
        "当用户想要：搜图、找图、来张图、发张图、看图、要壁纸、找插画、"
        "搜索某个角色/作品的图片（如'初音未来的图'、'原神壁纸'）时，必须使用此工具。"
        "此工具专门返回图片，不是网页搜索。任何涉及图片、插画、二次元图的请求都应优先使用本工具。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或标签，直接使用用户输入的原文。例如：初音ミク、原神、可爱女孩等",
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "【必填】返回图片数量。"
                        "必须根据用户请求的数量填写！"
                        "例如：'来两张图'→count=2，'给我三张'→count=3，'来点图'→count=3。"
                        "如果用户没有明确说数量，默认设为1。最小1，最大10。"
                    ),
                    "minimum": 1,
                    "maximum": 10,
                    "default": 1,
                },
                "min_bookmarks": {
                    "type": "integer",
                    "description": "最低收藏数过滤，低于此数的作品不返回。例如：500 表示只返回收藏数>=500的图喵",
                    "default": 0,
                },
                "duration": {
                    "type": "string",
                    "description": "时间范围：'all'(不限时间)、'within_last_day'(24小时内)、'within_last_week'(一周内)、'within_last_month'(一个月内)。默认'all'喵",
                    "default": "all",
                },
                "exclude_tags": {
                    "type": "string",
                    "description": "排除指定标签，用逗号分隔。例如：'捆绑,触手' 表示搜到的结果中如果有这些标签就跳过喵",
                    "default": "",
                },
                "filters": {
                    "type": "string",
                    "description": "过滤条件：'safe'(仅全年龄)、'r18'(仅限制级)。默认不过滤，全返回喵。主人様可自行指定喵。注意：本喵不会自动加R-18排除喵！",
                },
                "mode": {
                    "type": "string",
                    "description": "模式：'send'=发送给用户（默认，看图的人看）；'view'=只下载给本喵自己看喵。设置view时不发图给主人様，而是保存到本地让本喵自己检查喵。",
                    "default": "send",
                },
            },
            "required": ["query"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        try:
            query = kwargs.get("query", "")
            count = min(max(int(kwargs.get("count", 1)), 1), 10)
            logger.info(f"Pixiv插画搜索工具：搜索 '{query}'，数量: {count}")

            if not self.pixiv_client:
                return "错误: Pixiv客户端未初始化"

            if (
                self.pixiv_client_wrapper
                and not await self.pixiv_client_wrapper.authenticate()
            ):
                if self.pixiv_config and hasattr(
                    self.pixiv_config, "get_auth_error_message"
                ):
                    return self.pixiv_config.get_auth_error_message()
                return "Pixiv API 认证失败，请检查配置中的凭据信息。"

            tags = query.strip()
            mode = kwargs.get("mode", "send")
            min_bookmarks = int(kwargs.get("min_bookmarks", 0))
            duration = kwargs.get("duration", "all")
            exclude_tags = kwargs.get("exclude_tags", "")
            filters = kwargs.get("filters", "")
            return await self._search_illust(tags, query, context, count, mode, min_bookmarks, duration, exclude_tags, filters)

        except Exception as e:
            logger.error(f"Pixiv插画搜索失败: {e}")
            return f"搜索失败: {str(e)}"

    async def _search_illust(self, tags, query, context, count=1, mode="send", min_bookmarks=0, duration="all", exclude_tags="", filters=""):
        """按热度（收藏数）搜索插画，支持最低收藏数过滤、时间范围、排除标签和过滤条件喵"""
        import asyncio

        # 中文标签→日语标签转换喵 (提高热门搜索命中率)
        _tag_map = {
            "猫娘": "猫耳",
            "原神": "原神",
            "初音未来": "初音ミク",
            "可爱": "可愛い",
            "女孩": "女の子",
            "风景": "風景",
            "风景": "風景",
            "机甲": "メカ",
            "机甲": "メカ",
        }
        orig_tags = tags
        if tags in _tag_map:
            tags = _tag_map[tags]
            logger.info(f"标签翻译喵: {orig_tags} → {tags}")

        all_illusts = []
        page_count = 0
        next_params = None
        pages_to_fetch = 5

        while page_count < pages_to_fetch:
            try:
                if page_count == 0:
                    # 根据duration动态设置时间范围喵
                    kwargs_search = {
                        "search_target": "partial_match_for_tags",
                        "sort": "popular_desc",  # 按收藏数排序喵！
                        "filter": "for_ios",
                    }
                    if duration != "all":
                        kwargs_search["duration"] = duration
                    search_result = await asyncio.to_thread(
                        self.pixiv_client.search_illust, tags, **kwargs_search
                    )
                else:
                    if not next_params:
                        break
                    search_result = await asyncio.to_thread(
                        self.pixiv_client.search_illust, **next_params
                    )

                if not search_result or not hasattr(search_result, "illusts"):
                    break

                if search_result.illusts:
                    all_illusts.extend(search_result.illusts)
                    page_count += 1
                else:
                    break

                if hasattr(search_result, "next_url") and search_result.next_url:
                    next_params = self.pixiv_client.parse_qs(search_result.next_url)
                else:
                    break

                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"热度搜索第 {page_count + 1} 页出错: {e}")
                break

        if not all_illusts:
            return f"未找到关于 '{query}' 的插画。"

        sorted_illusts = sorted(
            all_illusts, key=lambda x: getattr(x, "total_bookmarks", 0), reverse=True
        )

        # 按最低收藏数过滤喵
        if min_bookmarks > 0:
            before = len(sorted_illusts)
            sorted_illusts = [ill for ill in sorted_illusts if getattr(ill, "total_bookmarks", 0) >= min_bookmarks]
            after = len(sorted_illusts)
            if after == 0:
                return f"收藏数 >= {min_bookmarks} 的作品太少了喵，试试降低 min_bookmarks 吧喵(ΦωФ;)✧"
            logger.info(f"min_bookmarks过滤: {before}→{after} 张喵")

        # 排除指定标签喵
        if exclude_tags:
            exclude_list = [t.strip() for t in exclude_tags.split(",") if t.strip()]
            before = len(sorted_illusts)
            sorted_illusts = [
                ill for ill in sorted_illusts
                if not any(
                    excl.lower() in [t.name.lower() if hasattr(t, "name") else str(t).lower() for t in getattr(ill, "tags", [])]
                    for excl in exclude_list
                )
            ]
            after = len(sorted_illusts)
            if after == 0:
                return f"排除标签 '{exclude_tags}' 后没有剩余作品了喵，试试放宽排除条件喵(ΦωФ;)✧"
            logger.info(f"排除标签 '{exclude_tags}' 过滤: {before}→{after} 张喵")

        # filters过滤条件喵（本喵不自动加也不默认过滤！主人様自己指定喵）
        if filters == "safe":
            before = len(sorted_illusts)
            sorted_illusts = [ill for ill in sorted_illusts if getattr(ill, "x_restrict", 0) == 0]
            after = len(sorted_illusts)
            if after == 0:
                return f"全年龄过滤后没有作品了喵(ΦωΦ;)✧"
            logger.info(f"filters=safe过滤: {before}→{after} 张喵")
        elif filters == "r18":
            before = len(sorted_illusts)
            sorted_illusts = [ill for ill in sorted_illusts if getattr(ill, "x_restrict", 0) > 0]
            after = len(sorted_illusts)
            if after == 0:
                return f"R-18过滤后没有作品了喵(ΦωΦ;)✧"
            logger.info(f"filters=r18过滤: {before}→{after} 张喵")

        event = self._get_event(context)
        if mode == "view":
            return await self._download_for_view(sorted_illusts, tags, count)
        elif event:
            return await self._send_pixiv_result(
                event, sorted_illusts, query, tags, count
            )
        else:
            return self._format_text_results(sorted_illusts, query, tags)

    async def _send_pixiv_result(self, event, items, query, tags, count=1):
        """发送按热度排序的结果"""
        logger.info(f"PixivIllustSearchTool: 准备发送 {count} 张图片")
        config = FilterConfig(
            r18_mode=self.pixiv_config.r18_mode if self.pixiv_config else "过滤 R18",
            filter_r18g_only=self.pixiv_config.filter_r18g_only
            if self.pixiv_config
            else False,
            ai_filter_mode=self.pixiv_config.ai_filter_mode
            if self.pixiv_config
            else "过滤 AI 作品",
            ai_detection_mode=self.pixiv_config.ai_detection_mode
            if self.pixiv_config
            else "field_or_tag",
            display_tag_str=f"搜索:{query}",
            return_count=count,
            logger=logger,
            show_filter_result=False,
            single_response_mode=self.pixiv_config.single_response_mode
            if self.pixiv_config
            else False,
            excluded_tags=[],
            forward_threshold=self.pixiv_config.forward_threshold
            if self.pixiv_config
            else False,
            show_details=self.pixiv_config.show_details if self.pixiv_config else True,
        )

        filtered_items, _ = filter_illusts_with_reason(items, config)
        if not filtered_items:
            return "找到插画但被过滤了 (可能是R18或AI作品)。"

        if not hasattr(event, "send"):
            return self._format_text_results(filtered_items, query, tags)

        expected_count = min(len(filtered_items), config.return_count)
        sent_batches = 0

        try:
            async for result in process_and_send_illusts_sorted(
                items,
                config,
                self.pixiv_client,
                event,
                build_detail_message,
                send_pixiv_image,
                send_forward_message,
                is_novel=False,
            ):
                try:
                    await event.send(result)
                    sent_batches += 1
                except Exception as e:
                    logger.warning(f"发送图片失败: {e}")

            if sent_batches > 0:
                mode = "转发消息" if config.forward_threshold else "普通消息"
                return (
                    f"🔥 找到了！为您发送了「{query}」一周内最热门的"
                    f" {expected_count} 张作品（{mode}）。"
                )

            return "找到插画但发送失败，请稍后再试。"
        except Exception as e:
            logger.error(f"发送失败: {e}")
            return "找到插画但发送过程中出现异常。"

    async def _download_for_view(self, illusts, tags, count=1):
        """下载图片到本地供本喵自己看喵，不发用户"""
        import asyncio
        import aiohttp
        import os
        from pathlib import Path

        save_dir = Path(__file__).parent.parent / "data" / "view_cache"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 先过滤（只保留符合配置的）
        from .tag import filter_illusts_with_reason, FilterConfig
        config = FilterConfig(
            r18_mode=self.pixiv_config.r18_mode if self.pixiv_config else "允许 R18",
            filter_r18g_only=self.pixiv_config.filter_r18g_only if self.pixiv_config else False,
            ai_filter_mode=self.pixiv_config.ai_filter_mode if self.pixiv_config else "显示 AI 作品",
            ai_detection_mode=self.pixiv_config.ai_detection_mode if self.pixiv_config else "field_or_tag",
            display_tag_str=f"搜索:{tags}",
            return_count=count,
            logger=logger,
            show_filter_result=False,
            single_response_mode=False,
            excluded_tags=[],
            forward_threshold=False,
            show_details=False,
        )
        filtered, _ = filter_illusts_with_reason(illusts, config)
        if not filtered:
            return f"搜到图但被过滤了喵 (R18/AI) 搜: {tags}"

        # 取前 count 张
        to_download = filtered[:count]
        results = []

        from .pixiv_utils import get_proxied_image_url

        async with aiohttp.ClientSession() as session:
            for i, ill in enumerate(to_download):
                # 获取图片URL
                url_obj = None
                if hasattr(ill, "meta_pages") and ill.meta_pages:
                    url_obj = ill.meta_pages[0].image_urls
                else:
                    class SinglePage:
                        pass
                    url_obj = SinglePage()
                    url_obj.original = getattr(ill.meta_single_page, "original_image_url", None) if hasattr(ill, "meta_single_page") else None
                    url_obj.large = getattr(ill.image_urls, "large", None) if hasattr(ill, "image_urls") else None
                    url_obj.medium = getattr(ill.image_urls, "medium", None) if hasattr(ill, "image_urls") else None

                img_url = url_obj.original or url_obj.large or url_obj.medium
                if not img_url:
                    continue

                proxied = get_proxied_image_url(img_url)

                try:
                    async with session.get(proxied, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            safe_name = f"pixiv_{ill.id}_p0.jpg"
                            save_path = save_dir / safe_name
                            with open(save_path, "wb") as f:
                                f.write(data)
                            results.append({
                                "id": ill.id,
                                "title": ill.title,
                                "user": ill.user.name if hasattr(ill, "user") else "未知",
                                "bookmarks": getattr(ill, "total_bookmarks", 0),
                                "path": str(save_path),
                                "size": len(data),
                            })
                except Exception as e:
                    logger.warning(f"下载图片 {ill.id} 失败: {e}")

        if not results:
            return f"下载失败喵，搜到{len(filtered)}张但都下不动喵"

        # 构建标签映射（用tag模块的解析函数喵）
        from .tag import _extract_tag_names
        tag_map = {}
        for ill in to_download:
            raw_tags = getattr(ill, "tags", None)
            if raw_tags is not None:
                tag_names = _extract_tag_names(raw_tags)
                tag_map[ill.id] = tag_names

        lines = []
        lines.append(f"📥 从Pixiv搜到 **{tags}** 喵！下载了 {len(results)} 张喵！")
        lines.append("")
        for r in results:
            size_kb = r["size"] / 1024
            lines.append(f"  [{r['id']}] **{r['title']}** by {r['user']} ({r['bookmarks']}⭐ {size_kb:.0f}KB)")
            # 显示标签喵
            tag_list = tag_map.get(r["id"], [])
            if tag_list:
                tag_str = "、".join(tag_list[:10])
                lines.append(f"  🏷️ {tag_str}")
            lines.append(f"  路径喵: `{r['path']}`")
        lines.append("")
        lines.append("本喵用 `astrbot_file_read_tool` 查看这些图喵！(ΦωФ)✧")
        return "\n".join(lines)

    def _get_event(self, context):
        try:
            agent_context = context.context if hasattr(context, "context") else context
            if hasattr(context, "event") and context.event:
                return context.event
            elif hasattr(agent_context, "event") and agent_context.event:
                return agent_context.event
        except Exception:
            pass
        return None

    def _format_text_results(self, items, query, tags):
        result = "找到以下插画:\n"
        for i, item in enumerate(items[:5], 1):
            title = getattr(item, "title", "未知标题")
            result += f"{i}. {title} (ID: {item.id})\n"
        return result


@dataclass
class PixivNovelSearchTool(FunctionTool[AstrAgentContext]):
    """
    Pixiv小说搜索工具
    """

    pixiv_client: Any = None
    pixiv_config: Any = None
    pixiv_client_wrapper: Any = None

    name: str = "pixiv_search_novel"
    description: str = "Pixiv小说搜索工具。用于搜索Pixiv上的小说，或者通过ID直接下载小说。支持输入关键词或纯数字ID。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或小说ID（纯数字）。",
                },
                "filters": {
                    "type": "string",
                    "description": "过滤条件，如 'safe', 'r18' 等",
                },
            },
            "required": ["query"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        try:
            query = kwargs.get("query", "")
            logger.info(f"Pixiv小说搜索工具：搜索 '{query}'")

            if not self.pixiv_client:
                return "错误: Pixiv客户端未初始化"

            if (
                self.pixiv_client_wrapper
                and not await self.pixiv_client_wrapper.authenticate()
            ):
                if self.pixiv_config and hasattr(
                    self.pixiv_config, "get_auth_error_message"
                ):
                    return self.pixiv_config.get_auth_error_message()
                return "Pixiv API 认证失败，请检查配置中的凭据信息。"

            tags = query.strip()
            return await self._search_novel(tags, query, context)

        except Exception as e:
            logger.error(f"Pixiv小说搜索失败: {e}")
            return f"搜索失败: {str(e)}"

    async def _search_novel(self, tags, query, context):
        import asyncio

        # ID 检查
        if query.isdigit():
            logger.info(f"检测到小说ID {query}")
            try:
                novel_detail = await asyncio.to_thread(
                    self.pixiv_client.novel_detail, int(query)
                )
                if novel_detail and novel_detail.novel:
                    event = self._get_event(context)
                    if event:
                        return await self._send_novel_result(
                            event, [novel_detail.novel], query, tags
                        )
                    else:
                        return f"找到小说: {novel_detail.novel.title} (ID: {query})，但无法发送文件(无事件上下文)。"
                else:
                    return f"未找到ID为 {query} 的小说。"
            except Exception as e:
                return f"获取小说详情失败: {str(e)}"

        # 标签搜索
        try:
            search_result = await asyncio.to_thread(
                self.pixiv_client.search_novel,
                tags,
                search_target="partial_match_for_tags",
            )

            if search_result and search_result.novels:
                event = self._get_event(context)
                if event:
                    return await self._send_novel_result(
                        event, search_result.novels, query, tags
                    )
                else:
                    return self._format_text_results(search_result.novels, query, tags)
            else:
                return f"未找到关于 '{query}' 的小说。"
        except Exception as e:
            return f"API调用错误: {str(e)}"

    async def _send_novel_result(self, event, items, query, tags):
        import asyncio

        if not items:
            return "未找到小说。"

        selected_item = items[0]  # 取第一个
        novel_id = str(selected_item.id)
        novel_title = selected_item.title

        logger.info(f"准备下载小说 {novel_title} (ID: {novel_id})")

        try:
            novel_content_result = await asyncio.to_thread(
                self.pixiv_client.webview_novel, novel_id
            )
            if not novel_content_result or not hasattr(novel_content_result, "text"):
                return f"无法获取小说内容 (ID: {novel_id})。"

            novel_text = novel_content_result.text

            try:
                pdf_bytes = await asyncio.to_thread(
                    self._create_pdf_from_text, novel_title, novel_text
                )
            except FileNotFoundError:
                return "无法生成PDF：字体文件丢失。"
            except Exception as e:
                return f"生成PDF失败: {str(e)}"

            # 加密
            password = hashlib.md5(novel_id.encode()).hexdigest()
            final_pdf_bytes = pdf_bytes
            password_notice = ""
            try:
                from PyPDF2 import PdfReader, PdfWriter

                reader = PdfReader(io.BytesIO(pdf_bytes))
                writer = PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                writer.encrypt(password)
                with io.BytesIO() as bs:
                    writer.write(bs)
                    final_pdf_bytes = bs.getvalue()
                password_notice = f"PDF已加密，密码: {password}"
            except Exception:
                password_notice = "PDF未加密。"

            # 发送
            safe_title = generate_safe_filename(novel_title, "novel")
            file_name = f"{safe_title}_{novel_id}.pdf"

            file_sent = False
            if event.get_platform_name() == "aiocqhttp" and event.get_group_id():
                try:
                    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                        AiocqhttpMessageEvent,
                    )

                    if isinstance(event, AiocqhttpMessageEvent):
                        client_bot = event.bot
                        group_id = event.get_group_id()
                        file_base64 = base64.b64encode(final_pdf_bytes).decode("utf-8")
                        await client_bot.upload_group_file(
                            group_id=group_id,
                            file=f"base64://{file_base64}",
                            name=file_name,
                        )
                        file_sent = True
                except Exception as e:
                    logger.error(f"群文件上传失败: {e}")

            author = (
                getattr(selected_item.user, "name", "未知作者")
                if hasattr(selected_item, "user")
                else "未知作者"
            )

            if file_sent:
                return f"已下载小说：\n**{novel_title}** - {author}\nID: {novel_id}\n文件已上传到群文件。\n{password_notice}\n(任务完成)"
            else:
                return f"已找到小说：\n**{novel_title}** - {author}\nID: {novel_id}\n无法发送文件，请尝试手动下载。\n(任务完成)"

        except Exception as e:
            logger.error(f"处理小说失败: {e}")
            return f"处理小说失败: {str(e)}"

    def _create_pdf_from_text(self, title: str, text: str) -> bytes:
        font_path = Path(__file__).parent.parent / "data" / "SmileySans-Oblique.ttf"
        if not font_path.exists():
            raise FileNotFoundError(f"字体文件不存在: {font_path}")

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("SmileySans", "", str(font_path), uni=True)
        pdf.set_font("SmileySans", size=20)
        pdf.multi_cell(0, 10, title, align="C")
        pdf.ln(10)
        pdf.set_font_size(12)
        pdf.multi_cell(0, 10, text)
        return pdf.output(dest="S")

    def _get_event(self, context):
        try:
            agent_context = context.context if hasattr(context, "context") else context
            if hasattr(context, "event") and context.event:
                return context.event
            elif hasattr(agent_context, "event") and agent_context.event:
                return agent_context.event
        except Exception:
            pass
        return None

    def _format_text_results(self, items, query, tags):
        result = "找到以下小说:\n"
        for i, item in enumerate(items[:5], 1):
            title = getattr(item, "title", "未知标题")
            result += f"{i}. {title} (ID: {item.id})\n"
        return result


@dataclass
class PixivRankingTool(FunctionTool[AstrAgentContext]):
    """
    Pixiv排行榜查询工具
    """

    pixiv_client: Any = None
    pixiv_config: Any = None
    pixiv_client_wrapper: Any = None

    name: str = "pixiv_ranking"
    description: str = (
        "【Pixiv排行榜查询专用工具】用于查看Pixiv上的官方排行榜（今日热门、本周热门等）。"
        "当用户想要：热榜、排行榜、今日热门、本周热门、Pixiv热榜时，必须使用此工具。"
        "此工具不按关键词搜索，而是返回Pixiv官方按热度排序的作品喵。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": (
                        "排行榜模式：'day'(今日)、'week'(本周)、'month'(本月)、"
                        "'day_male'(男性向)、'day_female'(女性向)、'week_original'(原创周榜)、"
                        "'week_rookie'(新人周榜)、'day_manga'(漫画日榜)、"
                        "'day_r18'(R18日榜)、'day_male_r18'(R18男性向)、'day_female_r18'(R18女性向)、"
                        "'week_r18'(R18周榜)、'week_r18g'(R18G周榜)。"
                        "默认'day'喵。"
                    ),
                    "default": "day",
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式 YYYY-MM-DD，可选喵。不传则获取最新排行榜喵。",
                    "default": "",
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "【必填】发送图片数量。"
                        "如果用户没说数量，默认设为1喵。最小1，最大10喵。"
                    ),
                    "minimum": 1,
                    "maximum": 10,
                    "default": 1,
                },
                "filters": {
                    "type": "string",
                    "description": "过滤条件：'safe'(仅全年龄)、'r18'(仅限制级)。默认不过滤，全返回喵。主人様可自行指定喵。注意：本喵不会自动加R-18排除喵！",
                    "default": "",
                },
            },
            "required": ["mode"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        try:
            mode = kwargs.get("mode", "day")
            date = kwargs.get("date", "") or None
            count = min(max(int(kwargs.get("count", 1)), 1), 10)
            filters = kwargs.get("filters", "")

            logger.info(f"Pixiv排行榜工具：模式 '{mode}'，日期: {date or '最新'}")

            if not self.pixiv_client:
                return "错误: Pixiv客户端未初始化"

            if (
                self.pixiv_client_wrapper
                and not await self.pixiv_client_wrapper.authenticate()
            ):
                if self.pixiv_config and hasattr(
                    self.pixiv_config, "get_auth_error_message"
                ):
                    return self.pixiv_config.get_auth_error_message()
                return "Pixiv API 认证失败，请检查配置中的凭据信息。"

            # 验证模式
            valid_modes = [
                "day", "week", "month",
                "day_male", "day_female", "week_original", "week_rookie", "day_manga",
                "day_r18", "day_male_r18", "day_female_r18", "week_r18", "week_r18g",
            ]
            if mode not in valid_modes:
                return f"无效的排行榜模式: {mode}喵。支持的: day, week, month, day_male, day_female, week_original, week_rookie, day_manga, day_r18, day_male_r18, day_female_r18, week_r18, week_r18g喵。"

            # 检查R18权限
            if "r18" in mode and self.pixiv_config and self.pixiv_config.r18_mode == "过滤 R18":
                return "当前R18模式设为「过滤 R18」，无法使用R18相关排行榜喵(ΦωΦ;)✧"
            if "r18g" in mode and self.pixiv_config and self.pixiv_config.filter_r18g_only:
                return "当前已开启「额外过滤R18G」，无法使用R18G相关排行榜喵(ΦωΦ;)✧"

            # 调用Pixiv API获取排行榜
            import asyncio
            ranking_result = await asyncio.to_thread(
                self.pixiv_client.illust_ranking, mode=mode, date=date
            )
            illusts = ranking_result.illusts if ranking_result and hasattr(ranking_result, "illusts") else []

            if not illusts:
                return f"未能获取到{date or '最新'}的{mode}排行榜数据喵(ΦωΦ;)✧"

            # 非manga模式过滤manga作品
            if "manga" not in mode.lower():
                before = len(illusts)
                illusts = [i for i in illusts if getattr(i, "type", None) != "manga"]
                after = len(illusts)
                if after == 0:
                    return f"{mode}排行榜结果均为漫画作品(manga)，已按非manga模式过滤喵(ΦωΦ;)✧"
                logger.info(f"排行榜 {mode} 已过滤 {before - after} 个漫画作品")

            # filters过滤
            if filters == "safe":
                before = len(illusts)
                illusts = [i for i in illusts if getattr(i, "x_restrict", 0) == 0]
                after = len(illusts)
                if after == 0:
                    return f"全年龄过滤后没有作品了喵(ΦωΦ;)✧"
                logger.info(f"filters=safe过滤: {before}→{after} 张")
            elif filters == "r18":
                before = len(illusts)
                illusts = [i for i in illusts if getattr(i, "x_restrict", 0) > 0]
                after = len(illusts)
                if after == 0:
                    return f"R-18过滤后没有作品了喵(ΦωΦ;)✧"
                logger.info(f"filters=r18过滤: {before}→{after} 张")

            event = self._get_event(context)
            if event:
                return await self._send_ranking_result(event, illusts, mode, count)
            else:
                return self._format_text_results(illusts, mode, "")

        except Exception as e:
            logger.error(f"Pixiv排行榜查询失败: {e}")
            return f"查询排行榜失败: {str(e)}"

    async def _send_ranking_result(self, event, items, mode, count=1):
        """发送排行榜结果"""
        from .tag import build_detail_message, FilterConfig, filter_illusts_with_reason, process_and_send_illusts_sorted

        logger.info(f"PixivRankingTool: 准备发送 {count} 张排行图片")
        config = FilterConfig(
            r18_mode=self.pixiv_config.r18_mode if self.pixiv_config else "过滤 R18",
            filter_r18g_only=self.pixiv_config.filter_r18g_only
            if self.pixiv_config else False,
            ai_filter_mode=self.pixiv_config.ai_filter_mode
            if self.pixiv_config else "过滤 AI 作品",
            ai_detection_mode=self.pixiv_config.ai_detection_mode
            if self.pixiv_config else "field_or_tag",
            display_tag_str=f"排行榜:{mode}",
            return_count=count,
            logger=logger,
            show_filter_result=False,
            single_response_mode=self.pixiv_config.single_response_mode
            if self.pixiv_config else False,
            excluded_tags=[],
            forward_threshold=self.pixiv_config.forward_threshold
            if self.pixiv_config else False,
            show_details=self.pixiv_config.show_details
            if self.pixiv_config else True,
        )

        filtered_items, _ = filter_illusts_with_reason(items, config)
        if not filtered_items:
            return f"获取到排行榜但被过滤了 (可能是R18/AI作品)喵(ΦωΦ;)✧"

        if not hasattr(event, "send"):
            return self._format_text_results(filtered_items, mode, "")

        expected_count = min(len(filtered_items), config.return_count)
        sent_batches = 0

        try:
            async for result in process_and_send_illusts_sorted(
                items,
                config,
                self.pixiv_client,
                event,
                build_detail_message,
                send_pixiv_image,
                send_forward_message,
                is_novel=False,
            ):
                try:
                    await event.send(result)
                    sent_batches += 1
                except Exception as e:
                    logger.warning(f"发送排行图片失败: {e}")

            if sent_batches > 0:
                mode_display = {
                    "day": "今日", "week": "本周", "month": "本月",
                    "day_male": "今日男性向", "day_female": "今日女性向",
                    "week_original": "本周原创", "week_rookie": "本周新人",
                    "day_manga": "今日漫画",
                    "day_r18": "今日R18", "day_male_r18": "今日R18男性向",
                    "day_female_r18": "今日R18女性向", "week_r18": "本周R18",
                    "week_r18g": "本周R18G",
                }.get(mode, mode)
                forward = "转发消息" if config.forward_threshold else "普通消息"
                return (
                    f"🔥 {mode_display}排行榜来了喵！"
                    f" 发送了 {expected_count} 张作品 ({forward})。"
                )
            return "获取排行榜成功但发送失败喵，请稍后再试喵(ΦωΦ;)✧"
        except Exception as e:
            logger.error(f"发送排行榜失败: {e}")
            return "获取排行榜成功但发送过程中出现异常喵(ΦωΦ;)✧"

    def _get_event(self, context):
        try:
            agent_context = context.context if hasattr(context, "context") else context
            if hasattr(context, "event") and context.event:
                return context.event
            elif hasattr(agent_context, "event") and agent_context.event:
                return agent_context.event
        except Exception:
            pass
        return None

    def _format_text_results(self, items, mode, _):
        result = f"找到{mode}排行榜作品:\n"
        for i, item in enumerate(items[:5], 1):
            title = getattr(item, "title", "未知标题")
            author = getattr(item.user, "name", "未知作者") if hasattr(item, "user") else "未知作者"
            result += f"{i}. {title} by {author} (ID: {item.id})\n"
        return result


def create_pixiv_llm_tools(
    pixiv_client=None, pixiv_config=None, pixiv_client_wrapper=None
) -> List[FunctionTool]:
    """
    创建Pixiv相关的LLM工具列表
    """
    logger.info(
        "创建Pixiv LLM工具，pixiv_client: %s, wrapper: %s"
        % (
            "已设置" if pixiv_client else "未设置",
            "已设置" if pixiv_client_wrapper else "未设置",
        )
    )

    tools = [
        PixivIllustSearchTool(
            pixiv_client=pixiv_client,
            pixiv_config=pixiv_config,
            pixiv_client_wrapper=pixiv_client_wrapper,
        ),
        PixivNovelSearchTool(
            pixiv_client=pixiv_client,
            pixiv_config=pixiv_config,
            pixiv_client_wrapper=pixiv_client_wrapper,
        ),
        PixivRankingTool(
            pixiv_client=pixiv_client,
            pixiv_config=pixiv_config,
            pixiv_client_wrapper=pixiv_client_wrapper,
        ),
    ]
    logger.info(f"已创建 {len(tools)} 个LLM工具")
    return tools
