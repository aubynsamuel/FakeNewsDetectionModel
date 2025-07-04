import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

sentence_transformer = SentenceTransformer("paraphrase-MiniLM-L12-v2")


def _semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculates semantic similarity using paraphrase-MiniLM-L12-v2 embeddings.
    Falls back to SpaCy/TF-IDF if paraphrase-MiniLM-L12-v2 is not loaded or fails.
    """
    if sentence_transformer:
        sentence_embeddings1 = sentence_transformer.encode(
            text1, show_progress_bar=False
        )
        sentence_embeddings2 = sentence_transformer.encode(
            text2, show_progress_bar=False
        )
        sentiment1 = TextBlob(text1).sentiment.polarity
        sentiment2 = TextBlob(text2).sentiment.polarity
        similarity = cosine_similarity(
            sentence_embeddings1.reshape(1, -1), sentence_embeddings2.reshape(1, -1)
        )[0][0]
        # Sentiment adjustment
        if sentiment1 * sentiment2 > 0:  # Both positive or both negative
            similarity = min(1.0, similarity * 1.1)
        elif sentiment1 * sentiment2 < 0:  # Opposing sentiment
            similarity = max(0.0, similarity * 0.9)
        return similarity


if __name__ == "__main__":
    claim = "House voting on rule to pave the way to final debate on Trump’s tax-and-spending bill – US politics live"
    content = """
    CBS News Live CBS News Live 2 Live Washington — House Republicans began taking a key procedural vote on President Trump's massive domestic policy bill Wednesday night, but it remains unclear if they have enough support to get the bill over the finish line. The vote continued into the overnight hours. Republican leadership and Mr. Trump spent much of the day Wednesday scrambling to shore up support from skeptical members ahead of a self-imposed July 4 deadline to get the bill — which squeaked through the Senate on Tuesday — to the president's desk. Before voting on final passage, the House needs to approve a resolution setting the rules of debate on the bill. After hours of delay, that crucial procedural vote began Wednesday at around 9:30 p.m. As of 1 o'clock Thursday morning, five House Republicans had voted no, which is theoretically enough for the rule vote to fail — but the vote is still open, and lawmakers can change from no to yes. Eight Republicans have not yet voted. Republicans can only afford three defections if all members are present and voting. Minutes before the vote began, Mr. Trump said on his Truth Social platform that the GOP caucus is "UNITED." But hours later, as a handful of Republican holdouts didn't appear to be budging, the president's mood seemed to sour. "What are the Republicans waiting for??? What are you trying to prove???" Mr. Trump wrote shortly after midnight. "MAGA IS NOT HAPPY, AND IT'S COSTING YOU VOTES!!!" He followed that up with a post saying, "FOR REPUBLICANS, THIS SHOULD BE AN EASY YES VOTE. RIDICULOUS!!!" While voting on the rule was underway, House Speaker Mike Johnson told Fox News' Sean Hannity he planned to keep the vote open "as long as it takes." The Louisiana Republican said he believes some lawmakers who voted no are "open for conversation" and their no votes are "placeholders" while they await answers to some questions about the bill. He said, "We believe we can get everybody to yes." "I'm absolutely confident we are going to land this plane and deliver for the American people," Johnson said. House GOP leaders had aimed to move ahead quickly on the signature legislation of Mr. Trump's second-term agenda, which includes ramped-up spending for border security, defense and energy production and extends trillions of dollars in tax cuts, partially offset by substantial cuts to health care and nutrition programs. But some House Republicans, who voted to pass an earlier version of the bill in May, are unhappy with the Senate's changes. Potential holdouts, including moderates and members of the conservative House Freedom Caucus, met with Mr. Trump on Wednesday as the White House put pressure on House Republicans to vote for the bill. One lawmaker called the meetings "very productive." But GOP 
    Rep. Andy Harris of Maryland, the chairman of the Freedom Caucus, told reporters earlier Wednesday that he expected the procedural vote to fail in the afternoon. In a possible sign of movement, one key Republican, Ohio Rep. Warren Davidson, announced on X Wednesday evening that he'd support the bill. It "isn't perfect, but it's the best we'll get," he wrote, adding that he would support the rule and final passage. Davidson was one of two Republicans who voted against the bill when the House first voted on the measure in May. The president kept up the pressure, posting on Truth Social about June's low border crossing statistics and adding, "All we need to do is keep it this way, which is exactly why Republicans need to pass "THE ONE, BIG, BEAUTIFUL BILL." Several members on both sides of the aisle had their flights canceled or delayed by bad weather as they raced 
    back to Washington for the vote, delaying the process. All the Democrats appeared to be on hand for proceedings by Wednesday 
    afternoon. House hardliners push back against Senate changes The House Rules Committee advanced the Senate's changes to the bill overnight, setting up the action on the floor. GOP Reps. Ralph Norman of South Carolina and Chip Roy of Texas joined Democrats on the panel to oppose the rule. Both are among the group of hardliners who are likely to oppose the procedural vote in the full House. "What the Senate did is unconscionable," Norman said. "I'll vote against it here and I'll vote against it on the floor until we get it right." Hours later, Norman returned to the Capitol following a meeting with Mr. Trump and other House Republicans. He described the meeting as "very productive" but didn't say whether he will ultimately vote yes, telling reporters he's still trying to learn more about how the bill will be implemented if it passes. Johnson has spent weeks pleading with his Senate counterparts not to make any major changes to the version of the bill that passed the lower chamber by a single vote in May. He said the Senate bill's changes "went a little further than many of us would've preferred." The Senate-passed bill includes steeper Medicaid cuts , a higher increase in the debt limit and changes to the House bill's green energy policies and the state and local tax deduction. Other controversial provisions that faced pushback in both chambers, including the sale of public lands in nearly a dozen states, a 10-year moratorium on states regulating artificial intelligence and an 
    excise tax on the renewable energy industry, were stripped from the Senate bill before heading back to the House. Johnson said Wednesday, before voting began, that "we are working through everybody's issues and making sure that we can secure this vote" amid the opposition. He added that he and the president are working to "convince everybody that this is the very best product that we can produce." "I feel good about where we are and where we're headed," Johnson added. Harris told reporters Wednesday that that the president should call the Senate back into town to come to an agreement on changes to the bill. GOP leaders, however, said the House would vote on the Senate bill "as-is." Should the House make changes to the bill, the revisions would require the Senate's approval, or force the two chambers to go to conference committee to iron out a final product that the two bodies could agree on, jeopardizing the bill's timely passage. Rep. Dusty Johnson, a South Dakota Republican, seemed optimistic after the White House meetings with holdouts Wednesday, saying "Donald Trump is a closer" and adding that "members 
    are moving to yes." "I know there are some members who think they're going to vote no right now," the South Dakota Republican said. "I think when the choice becomes failure or passage, they're going to understand that passage beats the hell out of failing." GOP Rep. Virginia Foxx of North Carolina likewise urged House Republicans to get the bill to the president's desk Wednesday. "President Trump has his pen in hand and is waiting for the House to complete its work," Foxx said. "We've championed this legislation for months, have guided it through the appropriate processes, and now we're on the one-yard line." Meanwhile, with few levers to combat the bill's passage, House Democrats spoke out forcefully against the legislation. "We will not stand by and watch Trump and his billionaire friends destroy this country without putting up one hell of a fight," Democratic 
    Rep. Jim McGovern of Massachusetts said, calling the bill a "massive betrayal of the American people." Jeffries said that "every single House Democrat will vote 'hell no' against this one, big ugly bill," while adding that "all we need are four House Republicans to join us in defense of their constituents who will suffer mightily from this bill." Democratic leaders called 
    out some Republicans by name, including Reps. Rob Bresnahan and Scott Perry of Pennsylvania and Reps. David Valadao and Young Kim of California. "It's unconscionable, it's unacceptable, it's un-American, and House Democrats are committing to you that we're going to do everything in our power to stop it," Jeffries said. "All we need are fou

    """

    # sentences = [str(sent) for sent in TextBlob(content).sentences]

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     similarity_scores = list(
    #         executor.map(partial(_semantic_similarity, claim), sentences)
    #     )

    # best_semantic_similarity = max(similarity_scores) if similarity_scores else 0.0

    verification_score = _semantic_similarity(claim, content)
    print(f"Verification score: {verification_score}")
    # print(f"Best semantic similarity score: {best_semantic_similarity}")
