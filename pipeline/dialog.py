from __future__ import annotations
from dataclasses import dataclass, field

FIXED_PRIORITY = [
    "material",
    "color",
    "size",
    "style",
    "budget",
    "use_case",
    "feature",
    "brand",
    "other",
]

SIMULATOR_AWARE_PRIORITY = [
    "other",
    "material",
    "color",
    "style",
    "budget",
    "use_case",
    "size",
    "feature",
]

QUESTION_TEMPLATES = {
    "material": "Do you have a preferred material?",
    "color": "Do you have a preferred color?",
    "size": "What size are you looking for?",
    "style": "Do you have a preferred style or fit?",
    "budget": "What budget range are you considering?",
    "use_case": "What will you mainly use this for?",
    "feature": "Are there any specific features you need?",
    "brand": "Do you have a preferred brand?",
    "other": "Is there anything else that matters to you?",
}

@dataclass
class ConversationState:
    session_id: str
    user_profile: dict
    current_turn: int = 0
    
    history: list[dict] = field(default_factory = list)
    known_constraints: list[str] = field(default_factory = list)
    asked_attributes: set[str] = field(default_factory = set)
    no_preference_attributes: set[str] = field(default_factory = set)
    superseded_constraints: set[str] = field(default_factory = set)
    override_detected: bool = False
    category: str | None = None
    initial_preference: str | None = None
    current_preference: str | None = None
    
class ConversationBrain:
    def __init__(self):
        self.states: dict[str, ConversationState] = {}
        
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.states[session_id] = ConversationState(
            session_id=session_id,
            user_profile=user_profile,
        )
    
    def get_state(self, session_id: str) -> ConversationState:
        if session_id not in self.states:
            raise RuntimeError(
                f"Unknown session: {session_id}"
            )
            
        return self.states[session_id]
    
    def record_asked_attribute(
        self,
        session_id: str,
        attribute: str | None,
    ) -> None:
        
        if attribute is None:
            return
        
        state = self.get_state(session_id)
        state.asked_attributes.add(attribute)
        
    def _extract_no_preference_attribute(
        self,
        user_message: str,
    ) -> str | None:
        
        prefix_1 = "I don't have an additional preference for "
        prefix_2 = "I don't have a preference for "
        
        if user_message.startswith(prefix_1):
            attribute = user_message[len(prefix_1):]
        elif user_message.startswith(prefix_2):
            attribute = user_message[len(prefix_2):]
            attribute = attribute.split(";", 1)[0]
        else:
            return None
        
        attribute = attribute.rstrip(".").strip()
        return attribute if attribute else None
        

    def _extract_revealed_constraints(
        self,
        user_message: str,
    ) -> list[str]:
        
        prefix = "For that, what matters is:"
        if not user_message.startswith(prefix):
            return []
        
        constraint_text = user_message[len(prefix):].strip()
        constraint_text = constraint_text.rstrip(".")
        
        constraints = [
            item.strip()
            for item in constraint_text.split(";")
            if item.strip()
        ]
        
        return constraints
    
    def _extract_category(
        self,
        user_message: str,
    ) -> str | None:
        prefix = "I'm looking for "
        if not user_message.startswith(prefix):
            return None
        
        remaining = user_message[len(prefix):]
        
        category = remaining.split(",", 1)[0]
        category = category.split(".", 1)[0]
        category = category.strip()
        
        return category if category else None
    
    def _extract_initial_requirement(
        self,
        user_message: str,
    ) -> str | None:

        marker = "A key requirement is:"

        if marker not in user_message:
            return None

        requirement = user_message.split(marker, 1)[1]
        requirement = requirement.strip().rstrip(".")
        return requirement if requirement else None
    
    def _extract_initial_preference(
        self,
        user_message: str,
    ) -> str | None:
        
        prefix = "I'm looking for"
        
        if not user_message.startswith(prefix):
            return None
        
        if "A key requirement is:" in user_message:
            return None
        
        if user_message.endswith("but I'm still exploring."):
            return None
        
        if "." not in user_message:
            return None
        
        preference = user_message.split(".", 1)[1]
        preference = preference.strip().rstrip(".")
        
        return preference if preference else None
    
    def _extract_override_value(
        self,
        user_message: str,
    ) -> str | None:
        
        marker = (
            "Actually, ignore my earlier preference. "
            "What I need is:"
        )
        
        if not user_message.startswith(marker):
            return None
        
        new_value = user_message[len(marker):]
        new_value = new_value.strip().rstrip(".")
        return new_value if new_value else None
    
    def choose_next_attribute(
        self,
        session_id: str,
    ) -> str | None:
        
        state = self.get_state(session_id)
        for attribute in ATTRIBUTE_PRIORITY:
            if attribute in state.asked_attributes:
                continue
            if attribute in state.no_preference_attributes:
                continue
            return attribute
        
        return None
    
    def question_for(
        self,
        attribute: str | None,
    ) -> str:

        if attribute is None:
            return "Here are my best matches so far."

        return QUESTION_TEMPLATES.get(
            attribute,
            "Could you tell me more about what you're looking for?",
        )
    
    
    def observe(self, session_id: str, user_message: str, turn: int,) -> None:
        state = self.get_state(session_id)
        state.current_turn = turn
        state.history.append({"turn": turn, "user": user_message})
        no_preference = self._extract_no_preference_attribute(user_message)
        if no_preference is not None:
            state.no_preference_attributes.add(no_preference)
        category = self._extract_category(user_message)
        if category:
            state.category = category
            
        initial_requirement = self._extract_initial_requirement(user_message)
        if (
            initial_requirement is not None
            and initial_requirement not in state.known_constraints 
        ):
            state.known_constraints.append(initial_requirement)
                  
        revealed = self._extract_revealed_constraints(
            user_message
        )
        
        for constraint in revealed:
            if constraint not in state.known_constraints:
                state.known_constraints.append(constraint)
                
        initial_preference = self._extract_initial_preference(user_message)
        if initial_preference is not None:
            state.initial_preference = initial_preference
            state.current_preference = initial_preference
            
            if initial_preference not in state.known_constraints:
                state.known_constraints.append(initial_preference)
                
        override_value = self._extract_override_value(user_message)
        if override_value is not None:
            state.override_detected = True
            
            old_preference = state.current_preference
            if old_preference is not None:
                if old_preference in state.known_constraints:
                    state.known_constraints.remove(old_preference)
                    
                state.superseded_constraints.add(old_preference)
                
            
            state.current_preference = override_value
            if override_value not in state.known_constraints:
                state.known_constraints.append(override_value)
                
    
    def choose_next_attribute(
        self,
        session_id: str,
        strategy: str = "fixed",
    ) -> str | None:

        state = self.get_state(session_id)

        if strategy == "simulator_aware":
            priority = SIMULATOR_AWARE_PRIORITY
        else:
            priority = FIXED_PRIORITY

        for attribute in priority:

            if attribute in state.asked_attributes:
                continue

            if attribute in state.no_preference_attributes:
                continue

            return attribute

        return None
    
