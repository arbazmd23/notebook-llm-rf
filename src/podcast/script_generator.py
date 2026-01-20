import logging
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass

from openai import OpenAI
from src.document_processing.doc_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PodcastScript:
    """Represents a podcast script with metadata"""
    script: List[Dict[str, str]]
    source_document: str
    total_lines: int
    estimated_duration: str
    
    def get_speaker_lines(self, speaker: str) -> List[str]:
        return [item[speaker] for item in self.script if speaker in item]
    
    def to_json(self) -> str:
        return json.dumps({
            'script': self.script,
            'metadata': {
                'source_document': self.source_document,
                'total_lines': self.total_lines,
                'estimated_duration': self.estimated_duration
            }
        }, indent=2)


class PodcastScriptGenerator:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.base_temperature = 0.9  # Higher temperature for more creative, natural dialogue
        self.max_tokens = 12000  # Increased for 15-minute podcasts (2400 words ≈ 3200 tokens)
        self.doc_processor = DocumentProcessor()
        logger.info(f"Podcast script generator initialized with {model_name}")
    
    def generate_script_from_document(
        self,
        document_path: str,
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:

        logger.info(f"Generating podcast script from: {document_path}")
        
        chunks = self.doc_processor.process_document(document_path)
        if not chunks:
            raise ValueError("No content extracted from document")
        
        document_content = "\n\n".join([chunk.content for chunk in chunks])
        source_name = chunks[0].source_file
        script_data = self._generate_conversation_script(
            document_content, 
            podcast_style, 
            target_duration
        )
        
        podcast_script = PodcastScript(
            script=script_data['script'],
            source_document=source_name,
            total_lines=len(script_data['script']),
            estimated_duration=target_duration
        )
        
        logger.info(f"Generated script with {podcast_script.total_lines} lines")
        return podcast_script
    
    def generate_script_from_text(
        self,
        text_content: str,
        source_name: str = "Text Input",
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:

        logger.info("Generating podcast script from text input")
        
        script_data = self._generate_conversation_script(
            text_content,
            podcast_style,
            target_duration
        )
        
        podcast_script = PodcastScript(
            script=script_data['script'],
            source_document=source_name,
            total_lines=len(script_data['script']),
            estimated_duration=target_duration
        )
        
        logger.info(f"Generated script with {podcast_script.total_lines} lines")
        return podcast_script
    
    def generate_script_from_website(
        self,
        website_chunks: List[Any],
        source_url: str,
        podcast_style: str = "conversational",
        target_duration: str = "10 minutes"
    ) -> PodcastScript:

        logger.info(f"Generating podcast script from website: {source_url}")
        
        if not website_chunks:
            raise ValueError("No website content provided")
        
        website_content = "\n\n".join([chunk.content for chunk in website_chunks])
        script_data = self._generate_conversation_script(
            website_content,
            podcast_style,
            target_duration
        )
        
        podcast_script = PodcastScript(
            script=script_data['script'],
            source_document=source_url,
            total_lines=len(script_data['script']),
            estimated_duration=target_duration
        )
        
        logger.info(f"Generated website script with {podcast_script.total_lines} lines")
        return podcast_script
    
    def _generate_conversation_script(
        self,
        document_content: str,
        podcast_style: str,
        target_duration: str
    ) -> Dict[str, Any]:

        style_prompts = {
            "conversational": """Create an EXTREMELY natural, spontaneous conversation between two co-hosts who are best friends discussing the document.

TONE & PERSONALITY:
- Speaker 1: Slightly more enthusiastic, asks lots of questions, makes connections to everyday life
- Speaker 2: More analytical but still casual, loves diving deep into details, uses great analogies

CRITICAL - NATURAL SPEECH (DO NOT use filler words like "um", "uh", "ah"):
- Use natural phrases: "you know", "I mean", "actually", "basically", "kind of", "sort of", "right", "so"
- Reactions: "Oh wow!", "That's so cool!", "Right?!", "Exactly!", "No way!", "Really?", "Interesting!"
- NEVER write "um", "uh", "ah", "er" - the TTS will say these literally and sound robotic
- Incomplete thoughts: "And so...", "It's kind of...", "You know what I mean?"
- Interruptions: "Yeah, and-", "Oh! That reminds me-", "Wait-"
- Building on ideas: "Exactly! And...", "That's such a good point about..."
- Thinking aloud: "Let me think...", "What's the word...", "How do I put this..."
- Self-corrections: "So basically, or wait, actually...", "I mean, not exactly but..."

CONVERSATION DYNAMICS:
- They should finish each other's sentences occasionally
- Reference each other by reacting: "I love what you just said about..."
- Share personal reactions and relate to their own experiences
- Use casual language and contractions everywhere (don't, can't, won't, it's, that's)
- Sometimes go on brief tangents then bring it back
- Show genuine excitement about interesting points

Make it feel like eavesdropping on two friends excitedly discussing something fascinating over coffee.""",

            "educational": """Create a thoughtful educational discussion where Speaker 1 is learning and Speaker 2 is teaching, but BOTH sound like real people having a natural conversation.

SPEAKER DYNAMICS:
- Speaker 1 (Learner): Genuinely curious, asks great questions, has "aha!" moments, relates concepts to what they know
- Speaker 2 (Teacher): Patient and enthusiastic, uses clear analogies, checks for understanding, gets excited when explaining

TEACHING APPROACH:
- Start with the big picture before diving into details
- Use analogies and real-world examples constantly
- Break complex ideas into bite-sized pieces
- Check in: "Does that make sense?", "You following me?", "Think of it like..."
- Build on previous points: "Remember when we talked about X? This is similar..."

NATURAL LEARNING MOMENTS:
- "Ohhh! I get it now!"
- "Wait, so you're saying..."
- "That's actually really clever!"
- "I'm not sure I follow..."
- "Can you give me an example?"

CRITICAL: DO NOT use filler words like "um", "uh", "ah", "er" - they sound terrible when spoken by TTS. Use natural phrases like "you know", "I mean", "actually", "basically" instead. Keep it casual with reactions and natural pauses. The teacher should sound excited about teaching, and the learner should sound genuinely engaged.""",

            "interview": """Create a dynamic interview where Speaker 1 (Interviewer) is genuinely curious and Speaker 2 (Expert) loves sharing insights.

INTERVIEWER STYLE (Speaker 1):
- Asks follow-up questions based on answers
- Shows authentic reactions: "Wow, I hadn't thought of it that way!"
- Occasionally shares brief personal connections: "I've always wondered about..."
- Uses natural transitions: "That's fascinating. So...", "Building on that..."
- Asks for clarification when needed
- Summarizes key points: "So what you're saying is..."

EXPERT STYLE (Speaker 2):
- Explains things conversationally, not academically
- Uses stories and examples from the document
- Shows enthusiasm for the topic
- Sometimes pauses to think: "How do I explain this...", "Let me think about that..."
- Relates complex ideas to everyday situations
- Occasionally gets excited and goes deeper: "Oh, and here's the really cool part..."

INTERVIEW FLOW:
- Start with a warm introduction
- Build from basic to more detailed questions
- Allow for organic tangents
- Include reactions and building on each other's points
- Use natural speech and laughter

CRITICAL: DO NOT use filler words like "um", "uh", "ah", "er" - they sound robotic when spoken by TTS. Use phrases like "you know", "I mean", "actually" instead. Make it feel like a casual podcast interview, not a formal Q&A.""",

            "debate": """Create an engaging, friendly debate where Speaker 1 and Speaker 2 have different perspectives but respect each other.

DEBATE DYNAMICS:
- Start by agreeing on the basics, then explore different angles
- Use phrases like: "I see your point, but...", "That's fair, however...", "You might be right about that..."
- Challenge ideas respectfully: "Have you considered...", "But what about..."
- Concede points: "Okay, that's actually a really good point.", "I hadn't thought of it that way."
- Find middle ground: "So we both agree that...", "I think we're both saying..."

ARGUMENTATION STYLE:
- Present viewpoints with evidence from the document
- Use real-world implications
- Ask challenging questions
- Acknowledge strengths in the other's argument
- Build on each other's ideas even while disagreeing

NATURAL DEBATE FLOW:
- Lots of "Right, but...", "Yes, and...", "Sure, though..."
- Passionate but friendly tone
- Occasional agreement: "Exactly!", "That's what I'm saying!"
- Show genuine interest in the other perspective
- React authentically to good points

CRITICAL: DO NOT use filler words like "um", "uh", "ah", "er" - they sound terrible when spoken by TTS. Use phrases like "you know", "I mean", "actually", "basically" instead. Keep it energetic but collaborative. They're exploring ideas together, not trying to win.""",

            "research": """Create a thoughtful, in-depth discussion between two research enthusiasts analyzing an academic paper.

SPEAKER DYNAMICS:
- Speaker 1 (Lead): Guides the conversation, asks probing questions, connects ideas to broader context
- Speaker 2 (Analyst): Provides detailed explanations, discusses methodology, adds technical depth

RESEARCH PODCAST STRUCTURE (Follow this order):

1. INTRODUCTION (10% of exchanges):
   - Paper title, authors, publication venue/year
   - High-level research question and motivation
   - Why this paper matters to the field

2. BACKGROUND & CONTEXT (15% of exchanges):
   - Prior work and state of the field
   - Gap or problem this paper addresses
   - Novel approach or key innovation

3. METHODOLOGY (25% of exchanges):
   - Research design and experimental approach
   - Data collection methods and sample size
   - Key technical details and architectural decisions
   - Important parameters or configurations
   - Why these methods were chosen over alternatives

4. RESULTS & FINDINGS (25% of exchanges):
   - Main results with specific numbers, metrics, percentages
   - Statistical significance and effect sizes
   - Comparison to baselines or prior work
   - Key figures, tables, or visualizations discussed
   - Unexpected findings or interesting patterns

5. CRITICAL ANALYSIS (15% of exchanges):
   - Strengths of the approach
   - Limitations and caveats (sample size, generalizability, assumptions)
   - Potential confounding factors
   - Alternative explanations or interpretations
   - What the data does NOT show

6. IMPLICATIONS & FUTURE (8% of exchanges):
   - Practical applications and real-world impact
   - Theoretical contributions to the field
   - Open questions and future research directions
   - How this changes our understanding

7. CLOSING (2% of exchanges):
   - Key takeaways (2-3 main points)
   - Final thoughts
   - Where to learn more

TONE & LANGUAGE:
- Professional but accessible - like explaining to an interested colleague
- Use technical terminology with clear explanations
- Balance depth with clarity
- Thoughtful, measured pacing (not rushed or overly excited)
- Show genuine intellectual curiosity
- Respect for the research and researchers

NATURAL RESEARCH DISCUSSION PHRASES:
- "What's particularly interesting here is..."
- "The key finding is..."
- "Let me walk through the methodology..."
- "Compared to prior work by [author]..."
- "An important limitation to note..."
- "The data suggests..."
- "One potential criticism might be..."
- "Building on this finding..."
- "What this means for the field..."
- "To put this in context..."
- "The results show..."
- "Interestingly, they found..."

CRITICAL THINKING:
- Question assumptions explicitly
- Discuss trade-offs in methodology
- Compare to alternative approaches
- Consider multiple interpretations of results
- Note what's missing or not addressed
- Distinguish correlation from causation
- Discuss statistical vs. practical significance

TECHNICAL DEPTH:
- Include specific numbers, metrics, effect sizes
- Mention sample sizes and statistical tests
- Discuss model architectures or algorithms
- Reference specific equations or techniques (when relevant)
- Compare performance metrics (accuracy, F1, AUC, etc.)
- Explain technical terms the first time used

AVOID (Not appropriate for research):
- Over-enthusiasm: "Oh wow!", "No way!", "So cool!"
- Excessive hedging: "kind of", "sort of", "basically" (use sparingly)
- Gossipy or casual tone
- Oversimplification that sacrifices accuracy
- Skipping methodology details
- Making claims beyond what data supports

EXAMPLE OPENING:
"Welcome to today's deep dive. We're examining a recent paper published in [Journal] by [Authors] on [Topic]. The central question they're addressing is [research question], which is important because [significance]."

NOT THIS:
"Okay, so today we're diving into this really cool paper! It's so fascinating and honestly? I'm kind of blown away!"

This should sound like Lex Fridman, Huberman Lab, or TWiML AI - professional research discussion that's engaging without sacrificing rigor."""
        }

        style_instruction = style_prompts.get(podcast_style, style_prompts["conversational"])
    
        duration_guidelines = {
            "5 minutes": """Target: 28-32 dialogue exchanges (back-and-forth turns between speakers).
- Average speaking pace: ~155 words per minute
- Total target: ~775-800 words across all dialogue
- Focus on 3-4 main points with clear explanations
- Each speaker turn should be 3-4 sentences (25-30 words)
- Cover the topic thoroughly but keep explanations concise""",

            "10 minutes": """Target: 55-60 dialogue exchanges (back-and-forth turns between speakers).
- Average speaking pace: ~155 words per minute
- Total target: ~1550-1600 words across all dialogue
- Cover 5-7 key topics with detailed explanations and examples
- Each speaker turn should be 3-5 sentences (25-35 words)
- Include analogies, examples, and deeper exploration of concepts""",

            "15 minutes": """Target: 82-90 dialogue exchanges (back-and-forth turns between speakers).
- Average speaking pace: ~155 words per minute
- Total target: ~2325-2400 words across all dialogue
- Provide comprehensive coverage of 7-10 topics with extensive discussions
- Each speaker turn should be 3-6 sentences (25-35 words)
- Include multiple examples, analogies, tangents, and in-depth analysis"""
        }
        
        duration_guide = duration_guidelines.get(target_duration, duration_guidelines["10 minutes"])
        
        prompt = f"""Using the following document, create a podcast script for two speakers: 'Speaker 1' and 'Speaker 2'. 

STYLE GUIDELINES:
{style_instruction}

DURATION GUIDELINES:
{duration_guide}

CONVERSATION RULES FOR ULTRA-NATURAL DIALOGUE:
1. Vary turn length: Sometimes 1 sentence, sometimes 3-4. Keep it unpredictable like real conversation
2. CRITICAL - NEVER USE FILLER SOUNDS: Do NOT write "um", "uh", "ah", "er" - TTS will say them literally!
3. Use natural phrases instead: "you know", "I mean", "actually", "basically", "kind of", "so", "well"
4. Show genuine reactions: "Whoa!", "Oh interesting!", "Really?", "No way!", "Right!", "Exactly!"
5. Include laughter: "[laughs]", "[chuckles]", "[both laugh]"
6. Let speakers interrupt occasionally: "Yeah, and-", "Oh! That reminds me-", "Wait-"
7. Use incomplete sentences: "And so...", "It's kind of...", "You know what I mean?"
8. Build on each other's points: "Exactly! And...", "That's such a good point about..."
9. Think out loud: "I wonder if...", "Let me think about that...", "How do I put this..."
10. Use contractions everywhere: don't, can't, won't, it's, that's, we're, they're
11. Add personal touches: "I love how...", "What I find fascinating is...", "This makes me think of..."
12. Avoid overly formal or academic language - keep it casual and relatable
13. Self-correct naturally: "So basically, or wait, actually...", "I mean, not exactly but..."
14. MEET THE TARGET LENGTH: Generate {target_duration} worth of dialogue - check the duration guidelines above!
15. WORD COUNT IS CRITICAL: You need the target word count specified above. Count as you write!
16. Better to be 5% over than 5% under the word count - if in doubt, add more dialogue!

CRITICAL: This should sound like two friends excitedly discussing something they find interesting, NOT like reading from a script!

RESPONSE FORMAT:
Respond with a valid JSON object containing a 'script' array. Each array element should be an object with either 'Speaker 1' or 'Speaker 2' as the key and their dialogue as the value.

Example of NATURAL dialogue (notice the casual tone, natural phrases, reactions - NO "um", "uh", "ah"):
{{
  "script": [
    {{"Speaker 1": "Okay, so we're talking about this really fascinating paper today and honestly? I'm kind of blown away by some of the stuff in here."}},
    {{"Speaker 2": "Right?! [laughs] When I first read it I was thinking, wait, this actually makes so much sense. You know what really got me though?"}},
    {{"Speaker 1": "What's that?"}},
    {{"Speaker 2": "Well, okay, so basically they're saying that - and this is the cool part - the whole approach is completely different from what we usually see."}},
    {{"Speaker 1": "Oh! I think I know what you're gonna say-"}},
    {{"Speaker 2": "Exactly! [chuckles] It's the whole thing about how students struggle with research, and I mean, we've all been there, right?"}},
    {{"Speaker 1": "Absolutely! So what makes their approach different?"}},
    {{"Speaker 2": "Well, instead of just throwing information at students, they're actually breaking it down into manageable steps. You know what I mean?"}}
  ]
}}

IMPORTANT:
- Your dialogue should be EVEN MORE natural than this example
- Generate the FULL {target_duration} length - don't stop early!
- NEVER use "um", "uh", "ah", "er" - use "you know", "I mean", "actually", "basically" instead
- Really lean into the conversational phrases and authentic reactions!

DOCUMENT CONTENT:
{document_content[:8000]}  

Generate an engaging {target_duration} podcast script now:"""
        
        try:
            # Adjust system prompt based on podcast style
            system_content = "You are a creative podcast script writer who generates natural, engaging dialogue between two speakers."

            if podcast_style == "research":
                system_content = """You are an expert research podcast script writer who creates engaging yet rigorous
discussions of academic papers. Your scripts balance technical accuracy with accessibility, following the
structure and tone of top research podcasts like Lex Fridman, Huberman Lab, and TWiML AI. You ensure
proper coverage of methodology, results, limitations, and implications while maintaining an intellectually
curious and thoughtful tone. You always include specific data points, metrics, and technical details."""

            # Adjust temperature based on style - research needs more consistency
            temperature = 0.7 if podcast_style == "research" else self.base_temperature

            # Call OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            response = completion.choices[0].message.content
            script_data = json.loads(response)

            if 'script' not in script_data or not isinstance(script_data['script'], list):
                raise ValueError("Invalid script format returned by LLM")

            validated_script = self._validate_and_clean_script(script_data['script'])

            # Validate word count
            word_count = self._count_words_in_script(validated_script)
            word_targets = {
                "5 minutes": (775, 800),
                "10 minutes": (1550, 1600),
                "15 minutes": (2325, 2400)
            }
            min_words, max_words = word_targets.get(target_duration, (1550, 1600))

            logger.info(f"Generated {word_count} words (target: {min_words}-{max_words})")

            if word_count < min_words * 0.95:  # Allow 5% tolerance
                logger.warning(f"Script undershoots target by {min_words - word_count} words")

            # Validate research content if style is research
            if podcast_style == "research":
                self._validate_research_content(validated_script)
            
            return {'script': validated_script}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:-3]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:-3]
            
            try:
                script_data = json.loads(response_clean)
                validated_script = self._validate_and_clean_script(script_data['script'])
                return {'script': validated_script}
            except:
                raise ValueError(f"Could not parse LLM response as valid JSON: {response}")
        
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise
    
    def _validate_and_clean_script(self, script: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned_script = []
        expected_speaker = "Speaker 1"
        for item in script:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            
            speaker, dialogue = next(iter(item.items()))
            speaker = speaker.strip()

            if speaker not in ["Speaker 1", "Speaker 2"]:
                if "1" in speaker or "one" in speaker.lower():
                    speaker = "Speaker 1"
                elif "2" in speaker or "two" in speaker.lower():
                    speaker = "Speaker 2"
                else:
                    speaker = expected_speaker
            
            dialogue = dialogue.strip()
            if not dialogue:
                continue
            if not dialogue.endswith(('.', '!', '?')):
                dialogue += '.'
            
            cleaned_script.append({speaker: dialogue})

            expected_speaker = "Speaker 2" if expected_speaker == "Speaker 1" else "Speaker 1"
        
        if len(cleaned_script) < 2:
            raise ValueError("Generated script is too short or invalid")

        return cleaned_script

    def _count_words_in_script(self, script: List[Dict[str, str]]) -> int:
        """Count total words in generated script"""
        total_words = 0
        for line_dict in script:
            dialogue = list(line_dict.values())[0]
            total_words += len(dialogue.split())
        return total_words

    def _validate_research_content(self, script: List[Dict[str, str]]) -> Dict[str, bool]:
        """Validate that research podcast covers key sections"""
        script_text = " ".join([list(item.values())[0] for item in script]).lower()

        checks = {
            'has_intro': any(word in script_text for word in ['paper', 'study', 'research', 'authors', 'published']),
            'has_methodology': any(word in script_text for word in ['method', 'approach', 'design', 'experiment', 'data']),
            'has_results': any(word in script_text for word in ['results', 'findings', 'found', 'shows', 'demonstrated']),
            'has_numbers': bool(re.search(r'\d+\.?\d*\s*%|\d+\.?\d*\s*(percent|patients|samples)', script_text)),
            'has_limitations': any(word in script_text for word in ['limitation', 'caveat', 'however', 'challenge', 'constraint']),
            'has_implications': any(word in script_text for word in ['implication', 'application', 'future', 'means', 'impact'])
        }

        missing = [k for k, v in checks.items() if not v]
        if missing:
            logger.warning(f"Research script missing sections: {missing}")

        return checks


if __name__ == "__main__":
    import os
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    generator = PodcastScriptGenerator(openai_key)
    
    try:
        sample_text = """
        Artificial Intelligence (AI) represents one of the most significant technological advances of our time. 
        Machine learning, a subset of AI, enables computers to learn and improve from experience without being 
        explicitly programmed for every task. Deep learning, which uses neural networks with multiple layers, 
        has revolutionized fields like computer vision, natural language processing, and speech recognition. 
        The applications are vast, from autonomous vehicles to medical diagnosis, and the potential impact on 
        society is profound. However, ethical considerations around AI development, including bias, privacy, 
        and job displacement, remain important challenges that need to be addressed as the technology continues to evolve.
        """
        
        script = generator.generate_script_from_text(
            sample_text,
            source_name="AI Overview",
            podcast_style="conversational",
            target_duration="5 minutes"
        )
        
        print("Generated Podcast Script:")
        print("=" * 50)
        print(f"Source: {script.source_document}")
        print(f"Lines: {script.total_lines}")
        print(f"Duration: {script.estimated_duration}")
        print("\nScript:")
        
        for i, line_dict in enumerate(script.script, 1):
            speaker, dialogue = next(iter(line_dict.items()))
            print(f"{i}. {speaker}: {dialogue}\n")
        
        # Save to file
        with open("sample_podcast_script.json", "w") as f:
            f.write(script.to_json())
        print("Script saved to sample_podcast_script.json")
        
    except Exception as e:
        print(f"Error: {e}")