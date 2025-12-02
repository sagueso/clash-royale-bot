"""Deck configuration with card properties and elixir costs."""

DECK = [
    {
        "name": "dark_prince",
        "url": "./ref_images/deck/dark_prince.png",
        "elixir_cost": 4,
        "type": "troop"
    },
    {
        "name": "prince",
        "url": "./ref_images/deck/prince.png",
        "elixir_cost": 5,
        "type": "troop"
    },
    {
        "name": "ice_wiz",
        "url": "./ref_images/deck/ice_wiz.png",
        "elixir_cost": 3,
        "type": "troop"
    },
    {
        "name": "knight",
        "url": "./ref_images/deck/knight.png",
        "elixir_cost": 3,
        "type": "troop"
    },
    {
        "name": "log",
        "url": "./ref_images/deck/log.png",
        "elixir_cost": 2,
        "type": "spell"
    },
    {
        "name": "mini_pekka",
        "url": "./ref_images/deck/mini_pekka.png",
        "elixir_cost": 4,
        "type": "troop"
    },
    {
        "name": "valk",
        "url": "./ref_images/deck/valk.png",
        "elixir_cost": 4,
        "type": "troop"
    },
    {
        "name": "musketeer",
        "url": "./ref_images/deck/musketeer.png",
        "elixir_cost": 4,
        "type": "troop"
    }
]


def get_card_by_name(card_name: str) -> dict:
    """Get card information by name.
    
    Args:
        card_name: Name of the card to retrieve
        
    Returns:
        Dictionary containing card information
        
    Raises:
        ValueError: If card name is not found in deck
    """
    for card in DECK:
        if card["name"] == card_name:
            return card
    raise ValueError(f"Card '{card_name}' not found in deck")


def get_card_elixir_cost(card_name: str) -> int:
    """Get elixir cost for a specific card.
    
    Args:
        card_name: Name of the card
        
    Returns:
        Elixir cost as integer
    """
    return get_card_by_name(card_name)["elixir_cost"]


def get_all_card_names() -> list:
    """Get list of all card names in the deck.
    
    Returns:
        List of card name strings
    """
    return [card["name"] for card in DECK]


def validate_deck() -> bool:
    """Validate that deck has exactly 8 cards with valid properties.
    
    Returns:
        True if deck is valid
        
    Raises:
        AssertionError: If deck validation fails
    """
    assert len(DECK) == 8, f"Deck must have 8 cards, found {len(DECK)}"
    
    for card in DECK:
        assert "name" in card, f"Card missing 'name' field: {card}"
        assert "url" in card, f"Card missing 'url' field: {card}"
        assert "elixir_cost" in card, f"Card missing 'elixir_cost' field: {card}"
        assert 1 <= card["elixir_cost"] <= 10, f"Invalid elixir cost for {card['name']}: {card['elixir_cost']}"
        assert "type" in card, f"Card missing 'type' field: {card}"
        assert card["type"] in ["troop", "spell", "building"], f"Invalid type for {card['name']}: {card['type']}"
    
    return True
