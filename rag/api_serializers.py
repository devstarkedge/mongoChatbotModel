from rest_framework import serializers


class AskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(min_length=2, max_length=500, trim_whitespace=True)
    session_id = serializers.RegexField(r"^[A-Fa-f0-9-]{1,64}$", required=False, allow_null=True)
    user_id = serializers.CharField(max_length=128, required=False, allow_blank=True, trim_whitespace=True)
    email = serializers.EmailField(max_length=254, required=False, allow_blank=True)
    role = serializers.CharField(max_length=64, required=False, allow_blank=True, trim_whitespace=True)
    top_k = serializers.IntegerField(min_value=1, max_value=50, required=False)
    hops = serializers.IntegerField(min_value=0, max_value=3, required=False)
    edit_index = serializers.IntegerField(min_value=0, required=False)
    collections = serializers.ListField(
        child=serializers.CharField(max_length=120), required=False, allow_empty=True,
    )


class SearchRequestSerializer(serializers.Serializer):
    query = serializers.CharField(min_length=2, max_length=500, trim_whitespace=True)
    user_id = serializers.CharField(max_length=128, required=False, allow_blank=True, trim_whitespace=True)
    email = serializers.EmailField(max_length=254, required=False, allow_blank=True)
    role = serializers.CharField(max_length=64, required=False, allow_blank=True, trim_whitespace=True)
    top_k = serializers.IntegerField(min_value=1, max_value=50, required=False)
    hops = serializers.IntegerField(min_value=0, max_value=3, required=False)
    collections = serializers.ListField(
        child=serializers.CharField(max_length=120), required=False, allow_empty=True,
    )
    hybrid = serializers.BooleanField(required=False)
    expand = serializers.BooleanField(required=False, default=False)


class RenameChatSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=120, trim_whitespace=True)


class IngestRequestSerializer(serializers.Serializer):
    collections = serializers.ListField(
        child=serializers.CharField(max_length=120), required=False, allow_empty=True,
    )
    full = serializers.BooleanField(required=False, default=False)
    max_docs = serializers.IntegerField(min_value=0, max_value=10_000_000, required=False)
