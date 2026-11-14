"""
Cookidoo MCP Server

Main server file containing MCP tool definitions for interacting with Cookidoo.
"""

import json
import re
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from cookidoo_service import CookidooService, load_cookidoo_settings
from guided import from_raw
from schemas import CustomRecipe, Step

PROMPTS_DIR = Path(__file__).parent
VERSION = tomllib.loads((PROMPTS_DIR / "pyproject.toml").read_text())["project"]["version"]
READ_ONLY = {"readOnlyHint": True, "openWorldHint": True}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True}

# Module-level authenticated session, created lazily on first use
_cookidoo_service: CookidooService | None = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if _cookidoo_service is not None:
            await _cookidoo_service.close()


mcp = FastMCP(
    "cookidoo-mcp-server",
    version=VERSION,
    instructions=(
        "Tools for the Thermomix Cookidoo platform: search recipes, read them, create "
        "or edit custom recipes, manage the shopping list, and plan meals in 'My Week'. "
        "Always show a custom recipe to the user and get explicit approval before creating or updating it."
    ),
    lifespan=lifespan,
)


def _get_service() -> CookidooService:
    global _cookidoo_service
    if _cookidoo_service is None:
        try:
            _cookidoo_service = CookidooService(load_cookidoo_settings())
        except ValueError as e:
            raise ToolError(f"Configuration error: {e}") from e
    return _cookidoo_service


def _steps(value: list[str | Step] | str) -> list[str | Step]:
    """Accept a list of plain or structured steps, or a string with one step per line."""
    if isinstance(value, str):
        return _lines(value, strip_numbering=True)
    steps: list[str | Step] = []
    for step in value:
        steps += [step] if isinstance(step, Step) else _lines([step], strip_numbering=True)
    return steps


def _lines(value: list[str] | str, strip_numbering: bool = False) -> list[str]:
    """Accept a list, or a string with one item per line (commas are kept)."""
    items = value.split("\n") if isinstance(value, str) else value
    if strip_numbering:
        items = [re.sub(r"^\s*(\d+[.)-]|[-•*])\s*", "", item) for item in items]
    return [item.strip() for item in items if item.strip()]


def _split_list(text: str) -> list[str]:
    """Split on newlines if present, otherwise on commas."""
    parts = text.split("\n") if "\n" in text else text.split(",")
    return [part.strip() for part in parts if part.strip()]


def _parse_recipe_json(recipe_json: str) -> CustomRecipe:
    try:
        return CustomRecipe.model_validate_json(recipe_json)
    except ValidationError as e:
        raise ToolError(f"Invalid recipe data: {e}") from e


def _minutes(seconds: int | None) -> str:
    return f"{round(seconds / 60)} min" if seconds else "?"


def _strip_markup(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


@mcp.tool(annotations=READ_ONLY)
async def connect_to_cookidoo() -> str:
    """
    Authenticate with Cookidoo using the credentials from the environment / .env file.

    Optional: every other tool connects automatically on first use. Call this to
    check the configuration or to force a fresh login.
    """
    service = _get_service()
    try:
        await service.login()
    except Exception as e:
        raise ToolError(f"Connection failed: {e}") from e
    return f"Connected to {service.localization_url} as {service.settings.email}"


@mcp.tool(annotations=READ_ONLY)
async def search_recipes(
    query: str = "",
    ingredients: str = "",
    max_total_time: int | None = None,
    max_results: int = 10,
    languages: str = "",
) -> str:
    """
    Search the Cookidoo recipe catalogue.

    Use this to find inspiration or recipes based on what is in the fridge, then pass
    the returned IDs to get_recipe_details.

    Args:
        query: Free-text query (e.g. "risotto", "quick soup")
        ingredients: Ingredients that must be used, comma- or newline-separated
        max_total_time: Maximum total time in minutes
        max_results: Maximum number of recipes to return (default: 10)
        languages: Language code(s) to filter by, comma-separated (e.g. "es", "en", "es,en").
            Defaults to the account's configured language.
    """
    if not (query or ingredients):
        raise ToolError("Provide a query and/or a list of ingredients.")
    try:
        result = await _get_service().search_recipes(
            query=query,
            ingredients=_split_list(ingredients),
            max_total_time=max_total_time,
            max_results=max_results,
            languages=languages or None,
        )
    except Exception as e:
        raise ToolError(f"Search failed: {e}") from e

    if not result.recipes:
        return "No recipes found. Try fewer or more common ingredients."
    lines = [f"Found {len(result.recipes)} recipes:", ""]
    for i, hit in enumerate(result.recipes, 1):
        lines.append(f"{i}. {hit.name}\n   ID: {hit.id} | {hit.url}")
    return "\n".join(lines)


@mcp.tool(annotations=READ_ONLY)
async def get_recipe_details(recipe_id: str) -> str:
    """
    Get the full details of a Cookidoo catalogue recipe: ingredients, steps, times.

    Args:
        recipe_id: The Cookidoo recipe ID (e.g. "r59322", "r907015")
    """
    try:
        recipe = await _get_service().get_recipe_details(recipe_id)
    except Exception as e:
        raise ToolError(f"Failed to get recipe details: {e}") from e

    lines = [
        f"Name: {recipe.name}",
        f"ID: {recipe.id}",
        f"Servings: {recipe.serving_size}",
        f"Active time: {_minutes(recipe.active_time)}",
        f"Total time: {_minutes(recipe.total_time)}",
        f"Difficulty: {recipe.difficulty}",
    ]
    if recipe.utensils:
        lines.append(f"Utensils: {', '.join(recipe.utensils)}")

    lines += ["", "Ingredients:"]
    for ingredient in recipe.ingredients:
        line = "  - " + " ".join(p for p in (ingredient.description, ingredient.name) if p)
        if ingredient.preparation:
            line += f", {ingredient.preparation}"
        lines.append(line)

    step_number = 0
    for group in recipe.step_groups:
        lines += ["", f"{group.title}:" if group.title else "Steps:"]
        for step in group.recipe_steps:
            step_number += 1
            lines.append(f"{step_number}. {_strip_markup(step.formatted_text)}")

    if recipe.notes:
        lines += ["", "Notes:", *(f"  - {_strip_markup(note)}" for note in recipe.notes)]
    lines += ["", f"URL: {recipe.url}"]
    return "\n".join(lines)


@mcp.tool(annotations=READ_ONLY)
async def get_shopping_list() -> str:
    """
    Get the current Cookidoo shopping list: scheduled/added recipes, ingredients, and additional items.
    """
    try:
        recipes, ingredients, additional = await _get_service().get_shopping_list()
    except Exception as e:
        raise ToolError(f"Failed to get shopping list: {e}") from e

    lines = []
    if recipes:
        lines.append(f"Recipes in Shopping List ({len(recipes)}):")
        for r in recipes:
            lines.append(f"- {r.name} (ID: {r.id} | {r.url})")
        lines.append("")

    if ingredients:
        lines.append(f"Ingredients ({len(ingredients)}):")
        for item in ingredients:
            check = "[x]" if item.is_owned else "[ ]"
            desc = f"{item.description}"
            if item.preparation:
                desc += f" ({item.preparation})"
            lines.append(f"- {check} {desc}")
        lines.append("")

    if additional:
        lines.append(f"Additional Items ({len(additional)}):")
        for item in additional:
            check = "[x]" if item.is_owned else "[ ]"
            lines.append(f"- {check} {item.name}")
        lines.append("")

    if not lines:
        return "Your Cookidoo shopping list is empty."
    return "\n".join(lines).strip()


@mcp.tool(annotations=WRITE)
async def add_to_shopping_list(recipes: list[str] | str) -> str:
    """
    Add recipes to the Cookidoo shopping list by their ID.
    Supports both catalogue recipe IDs (e.g. 'r55696') and custom recipe IDs (e.g. '01M7D2CG...').

    Args:
        recipes: Recipe ID or list of recipe IDs (comma- or newline-separated string, or list of strings)
    """
    recipe_ids = _lines(recipes) if isinstance(recipes, str) else [r.strip() for r in recipes if r.strip()]
    if not recipe_ids:
        raise ToolError("Provide at least one recipe ID to add to the shopping list.")
    try:
        added = await _get_service().add_recipes_to_shopping_list(recipe_ids)
    except Exception as e:
        raise ToolError(f"Failed to add recipes to shopping list: {e}") from e
    return (
        f"Successfully added {len(recipe_ids)} recipe(s) to your Cookidoo shopping list "
        f"({len(added)} ingredient items added)."
    )


@mcp.tool(annotations={**WRITE, "destructiveHint": True})
async def remove_from_shopping_list(recipes: list[str] | str) -> str:
    """
    Remove recipes from the Cookidoo shopping list by their ID.

    Args:
        recipes: Recipe ID or list of recipe IDs (comma- or newline-separated string, or list of strings)
    """
    recipe_ids = _lines(recipes) if isinstance(recipes, str) else [r.strip() for r in recipes if r.strip()]
    if not recipe_ids:
        raise ToolError("Provide at least one recipe ID to remove.")
    try:
        await _get_service().remove_recipes_from_shopping_list(recipe_ids)
    except Exception as e:
        raise ToolError(f"Failed to remove recipes from shopping list: {e}") from e
    return f"Removed {len(recipe_ids)} recipe(s) from your Cookidoo shopping list."


@mcp.tool(annotations=WRITE)
async def add_additional_items(items: list[str] | str) -> str:
    """
    Add standalone grocery items to the Cookidoo shopping list (e.g. 'milk', 'olive oil', 'paper towels').

    Args:
        items: Grocery item name or list of items (comma- or newline-separated string, or list of strings)
    """
    item_names = _lines(items) if isinstance(items, str) else [i.strip() for i in items if i.strip()]
    if not item_names:
        raise ToolError("Provide at least one item name to add.")
    try:
        added = await _get_service().add_additional_items(item_names)
    except Exception as e:
        raise ToolError(f"Failed to add additional items: {e}") from e
    return f"Added {len(added)} item(s) to your shopping list: {', '.join(item_names)}."


@mcp.tool(annotations={**WRITE, "destructiveHint": True})
async def clear_shopping_list() -> str:
    """
    Clear the entire Cookidoo shopping list (removes all recipes, ingredients and additional items).
    """
    try:
        await _get_service().clear_shopping_list()
    except Exception as e:
        raise ToolError(f"Failed to clear shopping list: {e}") from e
    return "Cookidoo shopping list cleared successfully."


@mcp.tool(annotations=READ_ONLY)
async def get_meal_plan(target_date: str = "") -> str:
    """
    Get the scheduled meals and recipes from Cookidoo's 'My Week' meal planner.

    Args:
        target_date: Optional reference date in YYYY-MM-DD format (defaults to current date/week)
    """
    if target_date:
        try:
            parsed_date = date.fromisoformat(target_date.strip())
        except ValueError:
            raise ToolError(f"Invalid date format '{target_date}'. Please use YYYY-MM-DD.")
    else:
        parsed_date = date.today()

    try:
        week_days = await _get_service().get_calendar_week(parsed_date)
    except Exception as e:
        raise ToolError(f"Failed to get meal plan: {e}") from e

    lines = [f"Meal Plan for the week of {parsed_date.isoformat()}:", ""]
    for day in week_days:
        lines.append(f"📅 {day.title or day.id}:")
        if day.recipes:
            for r in day.recipes:
                custom_tag = " [Custom]" if r.is_custom else ""
                time_str = f" ({_minutes(r.total_time)})" if r.total_time else ""
                lines.append(f"  - {r.name}{custom_tag}{time_str} | ID: {r.id} | {r.url}")
        else:
            lines.append("  (No recipes scheduled)")
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool(annotations=WRITE)
async def schedule_recipe(recipe_id: str, date_str: str) -> str:
    """
    Schedule a recipe to a specific date in Cookidoo's 'My Week' planner.
    Supports both catalogue recipe IDs ('r1234') and custom recipe IDs ('01...').

    Args:
        recipe_id: The Cookidoo recipe ID (catalogue or custom)
        date_str: Target date in YYYY-MM-DD format
    """
    try:
        target_date = date.fromisoformat(date_str.strip())
    except ValueError:
        raise ToolError(f"Invalid date format '{date_str}'. Please use YYYY-MM-DD.")

    try:
        await _get_service().schedule_recipe(target_date, recipe_id.strip())
    except Exception as e:
        raise ToolError(f"Failed to schedule recipe: {e}") from e
    return f"Recipe '{recipe_id}' successfully scheduled for {target_date.isoformat()}."


@mcp.tool(annotations={**WRITE, "destructiveHint": True})
async def unschedule_recipe(recipe_id: str, date_str: str) -> str:
    """
    Remove a scheduled recipe from a specific date in Cookidoo's 'My Week' planner.

    Args:
        recipe_id: The Cookidoo recipe ID (catalogue or custom)
        date_str: Target date in YYYY-MM-DD format
    """
    try:
        target_date = date.fromisoformat(date_str.strip())
    except ValueError:
        raise ToolError(f"Invalid date format '{date_str}'. Please use YYYY-MM-DD.")

    try:
        await _get_service().unschedule_recipe(target_date, recipe_id.strip())
    except Exception as e:
        raise ToolError(f"Failed to unschedule recipe: {e}") from e
    return f"Recipe '{recipe_id}' removed from {target_date.isoformat()}."


@mcp.tool(annotations=READ_ONLY)
async def list_custom_recipes() -> str:
    """List the custom recipes created on the user's Cookidoo account."""
    try:
        recipes = await _get_service().list_custom_recipes()
    except Exception as e:
        raise ToolError(f"Failed to list custom recipes: {e}") from e

    if not recipes:
        return "No custom recipes on this account yet."
    return "\n".join(
        f"- {r.name} ({r.serving_size} servings, {_minutes(r.total_time)})\n  ID: {r.id} | {r.url}"
        for r in recipes
    )


@mcp.tool(annotations=READ_ONLY)
async def get_custom_recipe(recipe_id: str) -> str:
    """
    Get one of the user's custom recipes as JSON, in the format expected by
    update_custom_recipe. Use it as the starting point before editing. Steps with
    machine settings or ingredient links come back as structured steps.

    Args:
        recipe_id: The custom recipe ID returned by list_custom_recipes
    """
    try:
        r, instructions = await _get_service().get_custom_recipe(recipe_id)
    except Exception as e:
        raise ToolError(f"Failed to get custom recipe: {e}") from e

    recipe = {
        "name": r.name,
        "ingredients": r.ingredients,
        "steps": [s if isinstance(s, str) else s.model_dump() for s in map(from_raw, instructions)],
        "servings": r.serving_size,
        "prep_time": max(1, round(r.active_time / 60)),
        "total_time": max(1, round(r.total_time / 60)),
        "hints": r.hints or None,
    }
    return f"URL: {r.url}\n\n{json.dumps(recipe, indent=2, ensure_ascii=False)}"


@mcp.tool(annotations=READ_ONLY)
async def generate_recipe_structure(
    name: str,
    ingredients: list[str] | str,
    steps: list[str | Step] | str,
    servings: int = 4,
    prep_time: int = 30,
    total_time: int = 60,
    hints: list[str] | str = "",
) -> str:
    """
    Generate and validate a recipe structure ready for upload to Cookidoo.

    Returns a JSON structure to show to the user and then pass to
    upload_custom_recipe or update_custom_recipe.

    Args:
        name: Recipe name (required)
        ingredients: Ingredients with quantities, as a list (or one per line)
        steps: Cooking steps, as a list (or one per line). A step without machine settings
            or ingredient links is plain text. Otherwise it is an object whose text holds one
            {settings} placeholder per entry of settings; the server writes each setting into
            the text and presets it on the Thermomix. A setting's "mode" picks its fields
            (allowed values are in the schema): "manual" (default: time, temperature, speed,
            reverse), "dough", "browning", "steaming", "turbo", "rice_cooker", "warm_up",
            "blend". "ingredients" links words of the text to an ingredient line. Example:
            {"text": "Add the onion and sauté {settings}.", "settings": [{"time": 180,
            "temperature": "120", "speed": "1"}], "ingredients": [{"text": "the onion",
            "ingredient": "1 onion"}]}
        servings: Number of servings (default: 4, range: 1-20)
        prep_time: Active preparation time in minutes (default: 30)
        total_time: Total time in minutes, at least prep_time (default: 60)
        hints: Optional cooking tips, as a list (or one per line)
    """
    try:
        recipe = CustomRecipe(
            name=name,
            ingredients=_lines(ingredients),
            steps=_steps(steps),
            servings=servings,
            prep_time=prep_time,
            total_time=total_time,
            hints=_lines(hints) or None,
        )
    except ValidationError as e:
        raise ToolError(f"Validation failed: {e}") from e
    return recipe.model_dump_json(indent=2)


@mcp.tool(annotations=WRITE)
async def upload_custom_recipe(recipe_json: str) -> str:
    """
    Create a brand new custom recipe on the user's Cookidoo account.

    Only call this after the user explicitly approved the recipe.

    Args:
        recipe_json: The validated recipe JSON from generate_recipe_structure
    """
    recipe = _parse_recipe_json(recipe_json)
    try:
        created = await _get_service().create_custom_recipe(recipe)
    except Exception as e:
        raise ToolError(f"Upload failed: {e}") from e
    return f"Recipe '{created.name}' created.\nRecipe ID: {created.id}\nURL: {created.url}"


@mcp.tool(annotations={**WRITE, "destructiveHint": True, "idempotentHint": True})
async def update_custom_recipe(recipe_id: str, recipe_json: str) -> str:
    """
    Replace the content of an existing custom recipe on Cookidoo.

    Get the current version with get_custom_recipe, edit it, show the result to the
    user and only call this after their explicit approval.

    Args:
        recipe_id: The custom recipe ID returned by list_custom_recipes
        recipe_json: The full, validated recipe JSON (same format as upload_custom_recipe)
    """
    recipe = _parse_recipe_json(recipe_json)
    try:
        updated = await _get_service().update_custom_recipe(recipe_id, recipe)
    except Exception as e:
        raise ToolError(f"Update failed: {e}") from e
    return f"Recipe '{updated.name}' updated.\nRecipe ID: {updated.id}\nURL: {updated.url}"


@mcp.prompt
def create_thermomix_recipe(dish: str, language: str = "en") -> str:
    """
    Guided workflow to write a Thermomix recipe for a dish and upload it to Cookidoo.

    Args:
        dish: The dish to create, e.g. "Creamy mushroom and parmesan risotto"
        language: "en" or "fr"
    """
    filename = "prompt_FR.md" if language.lower().startswith("fr") else "prompt.md"
    template = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    return re.sub(r"\[[^\]]*\]", f'"{dish}"', template, count=1)


if __name__ == "__main__":
    mcp.run()
