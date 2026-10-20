from pydantic import BaseModel, ConfigDict, model_validator


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    item_id: str


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    users: list[str]
    items: list[str]
    interactions: list[Interaction]

    @model_validator(mode="after")
    def validate_dataset(self):
        if len(set(self.users)) != len(self.users):
            raise ValueError("Duplicate users")
        if len(set(self.items)) != len(self.items):
            raise ValueError("Duplicate items")

        if any(not value for value in self.users + self.items):
            raise ValueError("Empty identifiers")

        users = set(self.users)
        items = set(self.items)
        seen = set()

        for event in self.interactions:
            if event.user_id not in users:
                raise ValueError("Unknown interaction user")
            if event.item_id not in items:
                raise ValueError("Unknown interaction item")

            pair = (event.user_id, event.item_id)
            if pair in seen:
                raise ValueError("Duplicate interaction")
            seen.add(pair)

        return self


class Recommendation(BaseModel):
    item_id: str
    score: int


class RecommendationResponse(BaseModel):
    user_id: str
    algorithm: str
    recommendations: list[Recommendation]
