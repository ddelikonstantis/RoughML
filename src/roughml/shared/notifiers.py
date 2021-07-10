import json
import logging
from abc import ABC

import emails
from emails.template import JinjaTemplate as _

from roughml.shared.configuration import Configuration

logger = logging.getLogger(__name__)


class Notifier(Configuration, ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class GmailNotifier(Notifier):
    def __init__(self, subject, email_template, user, password, sender):
        super().__init__(
            subject=subject,
            email_template=email_template,
            user=user,
            password=password,
            sender=sender,
        )

    def __call__(self, receivers, attachments, **context):
        logger.debug("Sending an email to %r", receivers)

        message = emails.html(
            subject=_(self.subject),
            html=_(self.email_template),
            mail_from=self.sender,
        )

        for attachment in attachments:
            message.attach(data=attachment.open("rb"), filename=attachment.name)

        response = message.send(
            to=receivers,
            render=context,
            smtp={
                "host": "smtp.gmail.com",
                "timeout": 5,
                "port": 587,
                "tls": True,
                **{"user": self.user, "password": self.password},
            },
        )

        if response.status_code != 250:
            raise ConnectionError(
                "Sending email failed with status code %d" % (response.status_code)
            )


class EndOfTrainingNotifier(GmailNotifier):
    _subject = "RoughML Flow {{identifier}} - {{generator}} / {{discriminator}} over dataset {{dataset.name}}"

    _email_template = """
        <h2>
            The <em>{{generator}} / {{discriminator}}</em> run over dataset <em>{{dataset.name}}</em>
            {% if succeeded %}
                completed <span style=\"color: green\">successfully</span>
            {% else %}
                <span style=\"color: red\">failed</span>
            {% endif %}
            after {{elapsed_time}} hours.
        </h2>
    """

    _sender = ("RoughML", "no-reply@roughml.com")

    def __init__(self, user, password):
        super().__init__(
            self._subject, self._email_template, user, password, self._sender
        )

    def __call__(self, receivers, log_file=None, **context):
        attachments = []
        if log_file is not None:
            attachments.append(log_file)

        super().__call__(receivers, attachments, **context)

    @classmethod
    def from_json(cls, credentials_path):
        with credentials_path.open("r") as file:
            data = json.load(file)

            return cls(data["user"], data["password"])
